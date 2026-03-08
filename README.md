# Shruti Parade

**Real-time piano transcription visualiser** — listens to audio (live mic, WAV/MP3/FLAC/OGG, or MIDI file), detects notes via dual-FFT harmonic-sieve + MPM pitch detection, and renders falling-note visuals with GPU-accelerated bloom and particle effects.

![Rust](https://img.shields.io/badge/Rust-2021_Edition-orange)
![License](https://img.shields.io/badge/license-MIT-blue)
[![CI](https://github.com/ProjectMalibulul/shruti-parade/actions/workflows/ci.yml/badge.svg)](https://github.com/ProjectMalibulul/shruti-parade/actions/workflows/ci.yml)

---

## Features

| Feature | Description |
|---------|-------------|
| **Dual-FFT harmonic-sieve detection** | Standard 4096-pt FFT for mid/treble + 8192-pt FFT for bass, with weighted harmonic template matching across all 88 piano keys (MIDI 21–108) |
| **MPM pitch refinement** | McLeod Pitch Method refines per-key confidence scores for improved accuracy |
| **Inharmonicity correction** | Stretched-partial compensation for realistic piano overtone series |
| **Three input modes** | Live microphone, audio files (WAV/MP3/FLAC/OGG via symphonia), MIDI files (via midly) |
| **GPU-accelerated rendering** | wgpu-powered falling-note visualiser with SDF rounded-rect notes, piano keyboard, and register-based colour palette |
| **Bloom post-processing** | Multi-pass Gaussian bloom (threshold → H-blur → V-blur → composite) with pulsing hit-line |
| **Particle effects** | Impact particles spawn at note onsets with gravity, drag, and fade |
| **Sustain pedal simulation** | Automatic pedal engagement detection with visual indicator bar |
| **Lock-free audio pipeline** | `rtrb` SPSC ring buffers for real-time audio → DSP data flow |
| **Multi-threaded architecture** | Separate threads for audio I/O, DSP, inference, MIDI routing, and rendering |
| **Transport controls** | Play/pause, seek (click/drag progress bar, arrow keys, Home/End) |
| **MIDI file synthesis** | Built-in multi-voice sine synthesiser for MIDI playback |

## Architecture

```
┌──────────────┐     rtrb ring      ┌──────────────┐   crossbeam    ┌──────────────┐
│  Audio I/O   │ ──── (f32) ──────▶ │     DSP      │ ── channel ──▶│  Inference   │
│  (cpal /     │                    │  (dual FFT + │  (PitchFrame) │  (onset /    │
│   symphonia) │                    │   MPM +      │               │   offset +   │
└──────────────┘                    │   sieve)     │               │   pedal sim) │
       │                            └──────────────┘               └──────┬───────┘
       │  rtrb ring                                                       │
       ▼                                                           NoteEvent channel
┌──────────────┐                                                          │
│  Playback    │                                                   ┌──────▼───────┐
│  (cpal out)  │                                                   │  MIDI Router │
└──────────────┘                                                   └──────┬───────┘
                                                                          │
                                                                   ┌──────▼───────┐
                                                                   │   Renderer   │
                                                                   │  (wgpu /     │
                                                                   │   winit)     │
                                                                   └──────────────┘
```

## Build

### Prerequisites

- **Rust** 1.70+ (2021 edition)
- **Linux**: `libasound2-dev`, `libwayland-dev`, `libxkbcommon-dev`
- **macOS / Windows**: No extra system dependencies

```bash
# Linux (Debian/Ubuntu)
sudo apt-get install -y libasound2-dev libwayland-dev libxkbcommon-dev

# Build
cargo build --release
```

### Run

```bash
# Live microphone input
cargo run --release

# Audio file (WAV, MP3, FLAC, OGG)
cargo run --release -- path/to/song.wav

# MIDI file
cargo run --release -- path/to/piece.mid
```

## Controls

| Key / Action | Description |
|---|---|
| **Space** | Play / Pause |
| **←** / **→** | Seek ±5 seconds |
| **Home** / **End** | Jump to start / end |
| **Click progress bar** | Seek to position |
| **Drag handle** | Scrub through audio |

## Configuration

Default settings are in [`src/config.rs`](src/config.rs):

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `audio` | `sample_rate` | 48000 | Audio sample rate (Hz) |
| `audio` | `buffer_frames` | 512 | Audio callback buffer size |
| `audio` | `ring_capacity` | 48000 | SPSC ring buffer capacity (samples) |
| `dsp` | `fft_size` | 4096 | Standard FFT window (mid/treble) |
| `dsp` | `bass_fft_size` | 8192 | Bass FFT window (MIDI 21–47) |
| `dsp` | `hop_size` | 512 | Hop between FFT frames |
| `inference` | `onset_threshold` | 0.5 | Note-on sensitivity (0–1, lower = more sensitive) |
| `inference` | `frame_threshold` | 0.3 | Note-off sensitivity |
| `render` | `width` × `height` | 1280 × 720 | Window size |
| `render` | `bloom_enabled` | true | GPU bloom post-processing |
| `render` | `particles_enabled` | true | Impact particle effects |

## CI / Release

- **CI** runs on every push/PR to `main`: formatting check, Clippy lints, build + test matrix across Ubuntu 22.04, Ubuntu 24.04, and macOS 13.
- **Release** workflow triggers on `v*` tags, builds release binaries for Linux (x86_64), Windows (x86_64), and macOS (aarch64), then uploads them as GitHub Release assets.

## Testing

```bash
# Run all tests (no GPU required)
cargo test --all-targets

# Run with output
cargo test --all-targets -- --nocapture

# Run only DSP tests
cargo test dsp_tests

# Run diagnostic pitch-detection tests
cargo test --test inference_diag -- --nocapture
```

## Project Structure

```
src/
├── main.rs          # Entry point, input mode detection, thread spawning
├── lib.rs           # Library re-exports for integration tests
├── audio.rs         # cpal audio capture & playback (real-time safe)
├── audio_file.rs    # symphonia-based audio file decoding & streaming
├── config.rs        # Configuration structs with defaults
├── dsp.rs           # Dual-FFT pipeline, harmonic sieve, MPM pitch refinement
├── events.rs        # Event types (NoteEvent, GPU instance structs)
├── inference.rs     # Onset/offset detector with sustain pedal simulation
├── midi_file.rs     # MIDI file parser + sine synthesiser
├── midi_router.rs   # Fan-out note events to renderer (+ future MIDI out)
├── particles.rs     # CPU-side particle system with gravity & drag
├── render.rs        # wgpu renderer (falling notes, keys, bloom, particles, UI)
├── timing.rs        # Lock-free AudioClock (atomic sample counter)
├── transport.rs     # Play/pause/seek state (lock-free atomics)
├── wav_ingest.rs    # Legacy hound-based WAV loader (unused, kept for reference)
└── shaders/
    ├── notes.wgsl   # Instanced geometry shader (notes, keys, particles)
    └── bloom.wgsl   # Fullscreen bloom passes + hit-line overlay
```

## Known Limitations

- **Low-pitch resolution**: Even with the 8192-pt bass FFT, notes below ~A1 (55 Hz) have limited frequency resolution. The MPM refinement helps but cannot fully overcome the Heisenberg-like time/frequency trade-off.
- **Inharmonicity model**: Uses a simplified inharmonicity coefficient. Real piano strings vary per-note; the current model uses a per-register approximation.
- **Sleep-based pacing**: Audio file streaming uses `thread::sleep` for real-time pacing, which is approximate. The ring buffer provides jitter tolerance.
- **No GPU text rendering**: FPS counter and transport status are shown as indicator bars rather than text labels.

## License

MIT License. See [LICENSE](LICENSE) for details.
