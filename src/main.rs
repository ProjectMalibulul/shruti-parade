mod audio;
mod audio_file;
mod config;
mod dsp;
mod events;
mod inference;
mod midi_file;
mod midi_router;
mod particles;
mod render;
mod timing;
mod wav_ingest;

use std::path::PathBuf;
use std::sync::Arc;

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

// ---------------------------------------------------------------------------
// Input mode detection
// ---------------------------------------------------------------------------

enum InputMode {
    AudioFile(PathBuf), // WAV, MP3, FLAC, OGG, …
    MidiFile(PathBuf),  // .mid / .midi
    LiveCapture,        // microphone
}

fn detect_mode(path: Option<PathBuf>) -> InputMode {
    match path {
        Some(p) => {
            let ext = p
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_lowercase())
                .unwrap_or_default();
            match ext.as_str() {
                "mid" | "midi" => InputMode::MidiFile(p),
                _ => InputMode::AudioFile(p), // symphonia will reject truly unsupported formats
            }
        }
        None => InputMode::LiveCapture,
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

    let file_path: Option<PathBuf> = std::env::args().nth(1).map(PathBuf::from);
    let mode = detect_mode(file_path);

    // ---- render channel (always needed) ----
    let (render_tx, render_rx) = crossbeam_channel::bounded::<NoteEvent>(256);

    // ---- shared audio clock ----
    let clock = Arc::new(AudioClock::new(config.audio.sample_rate));

    // ---- playback ring (4× capacity for jitter headroom) ----
    let playback_capacity = config.audio.ring_capacity * 4;
    let (playback_prod, playback_cons) = rtrb::RingBuffer::new(playback_capacity);

    // Keep handles alive so streams aren't dropped
    let _playback_handle: Option<AudioPlayback>;
    let _audio_handle: Option<AudioCapture>;

    match mode {
        // ══════════════════════════════════════════════════════════════════
        // MIDI file: parse → events straight to render, sine-synth playback
        // ══════════════════════════════════════════════════════════════════
        InputMode::MidiFile(path) => {
            info!("Loading MIDI file: {}", path.display());

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
                    ) {
                        tracing::error!("MIDI file error: {e:#}");
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

            // Playback at the file's native sample rate
            let file_sr = audio_file::audio_file_sample_rate(&path)?;
            info!("Audio file native sample rate: {file_sr} Hz");
            _playback_handle = Some(AudioPlayback::start(
                file_sr,
                config.audio.buffer_frames,
                playback_cons,
            )?);
            _audio_handle = None;

            // T0: audio file ingest
            let clock_af = clock.clone();
            let chunk = config.audio.buffer_frames;
            std::thread::Builder::new()
                .name("audio-ingest".into())
                .spawn(move || {
                    if let Err(e) = audio_file::stream_audio_file(
                        &path,
                        audio_prod,
                        Some(playback_prod),
                        clock_af,
                        chunk,
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
    render::run_render(config.render, render_rx, clock)?;

    info!("Shruti Parade — shutdown complete");
    Ok(())
}
