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
    info!(
        "Config: {}Hz buf={} ring={} FFT={}/{} hop={} mel={}({:.0}-{:.0}Hz) model={} ctx={}/{} render={}×{} fall={:.0}",
        config.audio.sample_rate, config.audio.buffer_frames, config.audio.ring_capacity,
        config.dsp.fft_size, config.dsp.bass_fft_size, config.dsp.hop_size,
        config.dsp.n_mels, config.dsp.mel_fmin, config.dsp.mel_fmax,
        config.inference.model_path, config.inference.context_frames, config.inference.overlap_frames,
        config.render.width, config.render.height, config.render.note_fall_speed,
    );

    let args: Vec<String> = std::env::args().skip(1).collect();
    let initial_mode = detect_mode(&args);

    let pending_file: Arc<Mutex<Option<PathBuf>>> = Arc::new(Mutex::new(None));

    // Spawn backend for initial mode
    let initial_handles = spawn_backend(&config, initial_mode)?;

    // Build the spawn closure that render.rs calls on file reload
    let config_for_spawn = config.clone();
    let spawn_fn: render::SpawnBackendFn = Box::new(move |path: Option<PathBuf>| {
        let mode = match path {
            Some(p) => detect_mode_from_path(p),
            None => InputMode::LiveCapture,
        };
        spawn_backend(&config_for_spawn, mode)
    });

    // Run render loop — blocks until window close (event loop never restarts)
    render::run_render(
        config.render.clone(),
        initial_handles,
        pending_file,
        spawn_fn,
    )?;

    info!("Shruti Parade — shutdown complete");
    Ok(())
}

// ---------------------------------------------------------------------------
// Backend spawning (used for initial load and hot-reload)
// ---------------------------------------------------------------------------

fn spawn_backend(config: &EngineConfig, mode: InputMode) -> Result<render::BackendHandles> {
    let (render_tx, render_rx) = crossbeam_channel::bounded::<NoteEvent>(256);
    let transport_state = Arc::new(TransportState::new());
    let (transport_tx, transport_rx) = crossbeam_channel::bounded(64);

    let playback_capacity = config.audio.ring_capacity * 4 * 2;
    let (playback_prod, playback_cons) = rtrb::RingBuffer::new(playback_capacity);

    let mut keepalive: Vec<Box<dyn std::any::Any>> = Vec::new();
    let clock: Arc<AudioClock>;

    match mode {
        InputMode::MidiFile(path) => {
            info!("Loading MIDI file: {}", path.display());
            clock = Arc::new(AudioClock::new(config.audio.sample_rate));

            let pb = AudioPlayback::start(
                config.audio.sample_rate,
                config.audio.buffer_frames,
                playback_cons,
                clock.clone(),
                transport_state.clone(),
            )?;
            keepalive.push(Box::new(pb));

            let clock_midi = clock.clone();
            let audio_cfg = config.audio.clone();
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
                        &audio_cfg,
                        transport_rx_thread,
                        transport_state_thread,
                    ) {
                        tracing::error!("MIDI file error: {e:#}");
                    }
                })?;
        }

        InputMode::AudioFileBasicPitch(path) => {
            info!("Transcribing with basic-pitch: {}", path.display());
            clock = Arc::new(AudioClock::new(config.audio.sample_rate));

            let tmp_dir = std::env::temp_dir().join("shruti-parade");
            let midi_path = basic_pitch::transcribe_to_midi(&path, &tmp_dir)?;
            info!("Using transcribed MIDI: {}", midi_path.display());

            let pb = AudioPlayback::start(
                config.audio.sample_rate,
                config.audio.buffer_frames,
                playback_cons,
                clock.clone(),
                transport_state.clone(),
            )?;
            keepalive.push(Box::new(pb));

            let clock_bp = clock.clone();
            let audio_cfg = config.audio.clone();
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
                        &audio_cfg,
                        transport_rx_thread,
                        transport_state_thread,
                    ) {
                        tracing::error!("basic-pitch MIDI playback error: {e:#}");
                    }
                })?;
        }

        InputMode::AudioFile(path) => {
            info!("Loading audio file: {}", path.display());

            let (audio_prod, audio_cons) = rtrb::RingBuffer::new(config.audio.ring_capacity);
            let (pitch_tx, pitch_rx) = crossbeam_channel::bounded::<PitchFrame>(64);
            let (event_tx, event_rx) = crossbeam_channel::bounded::<NoteEvent>(256);

            let file_sr = audio_file::audio_file_sample_rate(&path)?;
            info!("Audio file native sample rate: {file_sr} Hz");
            clock = Arc::new(AudioClock::new(file_sr));

            let pb = AudioPlayback::start(
                file_sr,
                config.audio.buffer_frames,
                playback_cons,
                clock.clone(),
                transport_state.clone(),
            )?;
            keepalive.push(Box::new(pb));

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

            let dsp_cfg = config.dsp.clone();
            std::thread::Builder::new()
                .name("dsp".into())
                .spawn(move || {
                    let mut pipe = DspPipeline::new(dsp_cfg, audio_cons, pitch_tx, file_sr);
                    pipe.run();
                })?;

            let inf_cfg = config.inference.clone();
            std::thread::Builder::new()
                .name("inference".into())
                .spawn(move || {
                    let mut eng = InferenceEngine::new(inf_cfg, pitch_rx, event_tx);
                    eng.run();
                })?;

            std::thread::Builder::new()
                .name("midi-router".into())
                .spawn(move || {
                    let router = MidiRouter::new(event_rx, render_tx);
                    router.run();
                })?;
        }

        InputMode::LiveCapture => {
            info!("Using live audio capture");
            clock = Arc::new(AudioClock::new(config.audio.sample_rate));

            let (audio_prod, audio_cons) = rtrb::RingBuffer::new(config.audio.ring_capacity);
            let (pitch_tx, pitch_rx) = crossbeam_channel::bounded::<PitchFrame>(64);
            let (event_tx, event_rx) = crossbeam_channel::bounded::<NoteEvent>(256);

            drop(playback_cons);
            drop(playback_prod);

            let cap = AudioCapture::start(&config.audio, clock.clone(), audio_prod)?;
            keepalive.push(Box::new(cap));

            let dsp_cfg = config.dsp.clone();
            let sr = config.audio.sample_rate;
            std::thread::Builder::new()
                .name("dsp".into())
                .spawn(move || {
                    let mut pipe = DspPipeline::new(dsp_cfg, audio_cons, pitch_tx, sr);
                    pipe.run();
                })?;

            let inf_cfg = config.inference.clone();
            std::thread::Builder::new()
                .name("inference".into())
                .spawn(move || {
                    let mut eng = InferenceEngine::new(inf_cfg, pitch_rx, event_tx);
                    eng.run();
                })?;

            std::thread::Builder::new()
                .name("midi-router".into())
                .spawn(move || {
                    let router = MidiRouter::new(event_rx, render_tx);
                    router.run();
                })?;
        }
    }

    Ok(render::BackendHandles {
        event_rx: render_rx,
        clock,
        transport_tx,
        transport_state,
        _keepalive: keepalive,
    })
}
