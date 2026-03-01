# Shruti Parade

**Real-time piano transcription visualiser** — listens to audio (live mic, WAV/MP3/FLAC/OGG, or MIDI file), detects notes via FFT-based harmonic-sieve pitch detection, and renders falling-note visuals with GPU-accelerated bloom and particle effects.

![Rust](https://img.shields.io/badge/Rust-2021_Edition-orange)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## Features

| Feature | Description |
|---------|-------------|
| **Harmonic-sieve pitch detection** | FFT → weighted harmonic template matching across all 88 piano keys (MIDI 21–108) |
| **Three input modes** | Live microphone, audio files (WAV/MP3/FLAC/OGG via symphonia), MIDI files (via midly) |
| **GPU-accelerated rendering** | wgpu-powered falling-note visualiser with SDF rounded-rect notes, hit-line, and piano keyboard |
| **Bloom post-processing** | Multi-pass Gaussian bloom (threshold → H-blur → V-blur → composite) |
| **Particle effects** | Impact particles spawn at note onsets with gravity, drag, and fade |
| **Lock-free audio pipeline** | `rtrb` SPSC ring buffers for real-time audio → DSP data flow |
| **Multi-threaded architecture** | Separate threads for audio I/O, DSP, inference, MIDI routing, and rendering |
| **MIDI file synthesis** | Built-in multi-voice sine synthesiser for MIDI playback |

## Architecture

```
┌──────────────┐     rtrb ring      ┌──────────────┐   crossbeam    ┌──────────────┐
│  Audio I/O   │ ──── (f32) ──────▶ │     DSP      │ ── channel ──▶│  Inference   │
│  (cpal /     │                    │  (FFT +      │  (PitchFrame) │  (onset /    │
│   symphonia) │                    │   harmonic   │               │   offset)    │
└──────────────┘                    │   sieve)     │               └──────┬───────┘
       │                            └──────────────┘                      │
       │  rtrb ring                                                NoteEvent channel
       ▼                                                                  │
┌──────────────┐                                                   ┌──────▼───────┐
│  Playback    │                                                   │  MIDI Router │
│  (cpal out)  │                                                   └──────┬───────┘
└──────────────┘                                                          │
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

## Configuration

Default settings are in [`src/config.rs`](src/config.rs):

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `audio` | `sample_rate` | 48000 | Audio sample rate (Hz) |
| `audio` | `buffer_frames` | 512 | Audio callback buffer size |
| `audio` | `ring_capacity` | 48000 | SPSC ring buffer capacity (samples) |
| `dsp` | `fft_size` | 4096 | FFT window size (power of 2) |
| `dsp` | `hop_size` | 512 | Hop between FFT frames |
| `inference` | `onset_threshold` | 0.5 | Note-on sensitivity (0–1, lower = more sensitive) |
| `inference` | `frame_threshold` | 0.3 | Note-off sensitivity |
| `render` | `width` × `height` | 1280 × 720 | Window size |
| `render` | `bloom_enabled` | true | GPU bloom post-processing |
| `render` | `particles_enabled` | true | Impact particle effects |

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
├── dsp.rs           # FFT pipeline, harmonic sieve, pitch energy computation
├── events.rs        # Event types (NoteEvent, GPU instance structs)
├── inference.rs     # Onset/offset detector on pitch energy frames
├── midi_file.rs     # MIDI file parser + sine synthesiser
├── midi_router.rs   # Fan-out note events to renderer (+ future MIDI out)
├── particles.rs     # CPU-side particle system with gravity & drag
├── render.rs        # wgpu renderer (falling notes, keys, bloom, particles)
├── timing.rs        # Lock-free AudioClock (atomic sample counter)
├── wav_ingest.rs    # Legacy hound-based WAV loader (unused, kept for reference)
└── shaders/
    ├── notes.wgsl   # Instanced geometry shader (notes, keys, particles)
    └── bloom.wgsl   # Fullscreen bloom passes + hit-line overlay
```

## Known Limitations

- **Low-pitch resolution**: At 48 kHz / 4096 FFT, frequency bin resolution is ~11.7 Hz. Notes below ~C2 (65 Hz) have only a few bins per fundamental. Increase `fft_size` to 8192 for better low-pitch accuracy at the cost of latency.
- **Inharmonicity**: The harmonic sieve assumes perfect integer harmonic ratios. Real piano strings exhibit slight inharmonicity (stretched partials), especially at extreme registers.
- **No sustain pedal**: Pedal events from MIDI files are parsed but not currently visualised.
- **Sleep-based pacing**: Audio file streaming uses `thread::sleep` for real-time pacing, which is approximate. The ring buffer provides jitter tolerance.

## License

MIT License. See [LICENSE](LICENSE) for details.
