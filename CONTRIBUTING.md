# Contributing to Shruti Parade

Thank you for your interest in contributing! This document outlines the development workflow and expectations.

## Development Setup

### Prerequisites

- **Rust** 1.70+ with `rustfmt` and `clippy` components
- **Linux**: `sudo apt-get install -y libasound2-dev libwayland-dev libxkbcommon-dev`
- **macOS / Windows**: No extra system dependencies

### Building

```bash
git clone <repo-url>
cd shruti-parade
cargo build
```

### Running Tests

```bash
# All tests (no GPU or audio device required)
cargo test --all-targets

# With diagnostic output
cargo test --all-targets -- --nocapture

# Specific test module
cargo test dsp_tests
cargo test dsp_edge_case_tests
cargo test --test inference_diag
```

## Code Quality

All PRs must pass CI checks:

### Formatting

```bash
cargo fmt --all -- --check
```

Fix formatting issues:

```bash
cargo fmt --all
```

### Linting

```bash
cargo clippy --all-targets -- -D warnings
```

### Tests

All existing tests must pass. New features should include tests.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed overview of the system design.

### Key Design Principles

1. **Real-time safety**: Audio callbacks must never allocate, lock, or syscall. Use `rtrb` for audio → DSP communication.
2. **Thread isolation**: Each pipeline stage runs on a dedicated thread. Communication is via bounded channels.
3. **Graceful degradation**: Dropped frames are acceptable for rendering. Use `try_send` for non-critical paths.
4. **Testability**: All DSP, timing, event, and config modules are testable without GPU or audio hardware.

### Module Ownership

| Module | Responsibility |
|--------|---------------|
| `audio.rs` | cpal stream setup (real-time callbacks) |
| `audio_file.rs` | symphonia decode + streaming |
| `dsp.rs` | FFT, harmonic sieve, pitch energies |
| `inference.rs` | Onset/offset detection |
| `midi_file.rs` | MIDI parse + sine synth |
| `render.rs` | wgpu pipelines, note tracking, bloom |
| `timing.rs` | Atomic audio clock |

## Pull Request Guidelines

1. **One concern per PR** — keep changes focused
2. **Include tests** for new functionality
3. **Update docs** if you change public APIs or architecture
4. **Run the full CI locally** before submitting:
   ```bash
   cargo fmt --all -- --check
   cargo clippy --all-targets -- -D warnings
   cargo test --all-targets
   ```

## Reporting Issues

When filing a bug report, please include:

- **OS and audio setup** (ALSA, PipeWire, PulseAudio, WASAPI, CoreAudio)
- **Input type** (live mic, audio file format, MIDI)
- **Rust version** (`rustc --version`)
- **Log output** (`RUST_LOG=shruti_parade=debug cargo run ...`)
- **Steps to reproduce**

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
