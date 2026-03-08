mod audio;
mod audio_file;
mod basic_pitch;
mod config;
mod dsp;
mod events;
mod inference;
mod midi_file;
mod midi_router;
mod particles;
mod render;
mod timing;
mod transport;
mod wav_ingest;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::Result;
use tracing::info;

use audio::AudioCapture;
use audio::AudioPlayback;
use config::EngineConfig;
use dsp::{DspPipeline, PitchFrame};
use events::NoteEvent;
use inference::InferenceEngine;
use midi_router::MidiRouter;
use timing::AudioClock;
use transport::TransportState;

// ---------------------------------------------------------------------------
// Input mode detection
// ---------------------------------------------------------------------------

enum InputMode {
    AudioFile(PathBuf),           // WAV, MP3, FLAC, OGG, …
    AudioFileBasicPitch(PathBuf), // Audio → basic-pitch MIDI transcription
    MidiFile(PathBuf),            // .mid / .midi
    LiveCapture,                  // microphone
}

fn detect_mode(args: &[String]) -> InputMode {
    let use_basic_pitch = args.iter().any(|a| a == "--basic-pitch");
    let path = args.iter().find(|a| !a.starts_with('-')).map(PathBuf::from);

    match path {
        Some(p) => {
            let ext = p
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase())
                .unwrap_or_default();
            match ext.as_str() {
                "mid" | "midi" => InputMode::MidiFile(p),
                _ if use_basic_pitch => InputMode::AudioFileBasicPitch(p),
                _ => InputMode::AudioFile(p),
            }
        }
        None => InputMode::LiveCapture,
    }
}

fn detect_mode_from_path(path: PathBuf) -> InputMode {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "mid" | "midi" => InputMode::MidiFile(path),
        _ => InputMode::AudioFile(path),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    // ---- logging ----
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "shruti_parade=debug,wgpu=warn".into()),
        )
        .init();

    info!("Shruti Parade — starting up");

    let config = EngineConfig::default();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut mode = detect_mode(&args);

    let pending_file: Arc<Mutex<Option<PathBuf>>> = Arc::new(Mutex::new(None));

    loop {
        // ---- render channel (always needed) ----
        let (render_tx, render_rx) = crossbeam_channel::bounded::<NoteEvent>(256);

        // ---- transport controls ----
        let transport_state = Arc::new(TransportState::new());
        let (transport_tx, transport_rx) = crossbeam_channel::bounded(64);

        // ---- playback ring (4× capacity for jitter headroom) ----
        let playback_capacity = config.audio.ring_capacity * 4;
        let (playback_prod, playback_cons) = rtrb::RingBuffer::new(playback_capacity);

        // Keep handles alive so streams aren't dropped
        let _playback_handle: Option<AudioPlayback>;
        let _audio_handle: Option<AudioCapture>;

        // ---- shared audio clock (created per-mode with correct sample rate) ----
        let clock: Arc<AudioClock>;

        match mode {
            // ══════════════════════════════════════════════════════════════════
            // MIDI file: parse → events straight to render, sine-synth playback
            // ══════════════════════════════════════════════════════════════════
            InputMode::MidiFile(path) => {
                info!("Loading MIDI file: {}", path.display());

                // Synth is digital — clock and playback both run at the engine SR
                clock = Arc::new(AudioClock::new(config.audio.sample_rate));

                // Start cpal output at the engine's sample rate (synth is digital)
                _playback_handle = Some(AudioPlayback::start(
                    config.audio.sample_rate,
                    config.audio.buffer_frames,
                    playback_cons,
                )?);
                _audio_handle = None;

                let clock_midi = clock.clone();
                let sr = config.audio.sample_rate;
                let chunk = config.audio.buffer_frames;
                let transport_state_thread = transport_state.clone();
                let transport_rx_thread = transport_rx.clone();
                std::thread::Builder::new()
                    .name("midi-file".into())
                    .spawn(move || {
                        if let Err(e) = midi_file::stream_midi_file(
                            &path,
                            render_tx,
                            Some(playback_prod),
                            clock_midi,
                            sr,
                            chunk,
                            transport_rx_thread,
                            transport_state_thread,
                        ) {
                            tracing::error!("MIDI file error: {e:#}");
                        }
                    })?;
            }

            // ══════════════════════════════════════════════════════════════════
            // Audio file + basic-pitch: transcribe to MIDI, then play as MIDI
            // ══════════════════════════════════════════════════════════════════
            InputMode::AudioFileBasicPitch(path) => {
                info!("Transcribing with basic-pitch: {}", path.display());

                // Synth is digital — clock and playback both run at the engine SR
                clock = Arc::new(AudioClock::new(config.audio.sample_rate));

                let tmp_dir = std::env::temp_dir().join("shruti-parade");
                let midi_path = basic_pitch::transcribe_to_midi(&path, &tmp_dir)?;
                info!("Using transcribed MIDI: {}", midi_path.display());

                _playback_handle = Some(AudioPlayback::start(
                    config.audio.sample_rate,
                    config.audio.buffer_frames,
                    playback_cons,
                )?);
                _audio_handle = None;

                let clock_bp = clock.clone();
                let sr = config.audio.sample_rate;
                let chunk = config.audio.buffer_frames;
                let transport_state_thread = transport_state.clone();
                let transport_rx_thread = transport_rx.clone();
                std::thread::Builder::new()
                    .name("basic-pitch-midi".into())
                    .spawn(move || {
                        if let Err(e) = midi_file::stream_midi_file(
                            &midi_path,
                            render_tx,
                            Some(playback_prod),
                            clock_bp,
                            sr,
                            chunk,
                            transport_rx_thread,
                            transport_state_thread,
                        ) {
                            tracing::error!("basic-pitch MIDI playback error: {e:#}");
                        }
                    })?;
            }

            // ══════════════════════════════════════════════════════════════════
            // Audio file (WAV/MP3/FLAC/OGG): full DSP → inference pipeline
            // ══════════════════════════════════════════════════════════════════
            InputMode::AudioFile(path) => {
                info!("Loading audio file: {}", path.display());

                let (audio_prod, audio_cons) = rtrb::RingBuffer::new(config.audio.ring_capacity);
                let (pitch_tx, pitch_rx) = crossbeam_channel::bounded::<PitchFrame>(64);
                let (event_tx, event_rx) = crossbeam_channel::bounded::<NoteEvent>(256);

                // Clock must match the file's native SR so that
                // now_seconds() = samples / file_sr = correct wall-clock time.
                let file_sr = audio_file::audio_file_sample_rate(&path)?;
                info!("Audio file native sample rate: {file_sr} Hz");
                clock = Arc::new(AudioClock::new(file_sr));
                _playback_handle = Some(AudioPlayback::start(
                    file_sr,
                    config.audio.buffer_frames,
                    playback_cons,
                )?);
                _audio_handle = None;

                // T0: audio file ingest
                let clock_af = clock.clone();
                let chunk = config.audio.buffer_frames;
                let transport_state_thread = transport_state.clone();
                let transport_rx_thread = transport_rx.clone();
                std::thread::Builder::new()
                    .name("audio-ingest".into())
                    .spawn(move || {
                        if let Err(e) = audio_file::stream_audio_file(
                            &path,
                            audio_prod,
                            Some(playback_prod),
                            clock_af,
                            chunk,
                            transport_rx_thread,
                            transport_state_thread,
                        ) {
                            tracing::error!("Audio file error: {e:#}");
                        }
                    })?;

                // T1: DSP — use the file's actual sample rate for correct frequency mapping
                let dsp_cfg = config.dsp.clone();
                std::thread::Builder::new()
                    .name("dsp".into())
                    .spawn(move || {
                        let mut pipe = DspPipeline::new(dsp_cfg, audio_cons, pitch_tx, file_sr);
                        pipe.run();
                    })?;

                // T2: inference
                let inf_cfg = config.inference.clone();
                std::thread::Builder::new()
                    .name("inference".into())
                    .spawn(move || {
                        let mut eng = InferenceEngine::new(inf_cfg, pitch_rx, event_tx);
                        eng.run();
                    })?;

                // T3: MIDI router
                std::thread::Builder::new()
                    .name("midi-router".into())
                    .spawn(move || {
                        let router = MidiRouter::new(event_rx, render_tx);
                        router.run();
                    })?;
            }

            // ══════════════════════════════════════════════════════════════════
            // Live microphone capture: full DSP → inference pipeline
            // ══════════════════════════════════════════════════════════════════
            InputMode::LiveCapture => {
                info!("Using live audio capture");

                // Live capture uses the configured engine SR
                clock = Arc::new(AudioClock::new(config.audio.sample_rate));

                let (audio_prod, audio_cons) = rtrb::RingBuffer::new(config.audio.ring_capacity);
                let (pitch_tx, pitch_rx) = crossbeam_channel::bounded::<PitchFrame>(64);
                let (event_tx, event_rx) = crossbeam_channel::bounded::<NoteEvent>(256);

                _playback_handle = None;
                drop(playback_cons);
                drop(playback_prod);

                _audio_handle = Some(AudioCapture::start(
                    &config.audio,
                    clock.clone(),
                    audio_prod,
                )?);

                // T1: DSP — live capture uses the configured sample rate
                let dsp_cfg = config.dsp.clone();
                let sr = config.audio.sample_rate;
                std::thread::Builder::new()
                    .name("dsp".into())
                    .spawn(move || {
                        let mut pipe = DspPipeline::new(dsp_cfg, audio_cons, pitch_tx, sr);
                        pipe.run();
                    })?;

                // T2: inference
                let inf_cfg = config.inference.clone();
                std::thread::Builder::new()
                    .name("inference".into())
                    .spawn(move || {
                        let mut eng = InferenceEngine::new(inf_cfg, pitch_rx, event_tx);
                        eng.run();
                    })?;

                // T3: MIDI router
                std::thread::Builder::new()
                    .name("midi-router".into())
                    .spawn(move || {
                        let router = MidiRouter::new(event_rx, render_tx);
                        router.run();
                    })?;
            }
        }

        // ---- T4: render loop (blocks main thread) ----
        render::run_render(
            config.render.clone(),
            render_rx,
            clock,
            transport_tx,
            transport_state,
            pending_file.clone(),
        )?;

        // Check if the user dropped / picked a new file
        let new_file = pending_file.lock().unwrap().take();
        if let Some(path) = new_file {
            info!("Reloading with new file: {}", path.display());
            mode = detect_mode_from_path(path);
            // Pipeline threads exit naturally when channels/rings are dropped
            continue;
        }

        break;
    } // end loop

    info!("Shruti Parade — shutdown complete");
    Ok(())
}
