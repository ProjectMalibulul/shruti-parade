# Architecture

This document describes the internal architecture of Shruti Parade, a real-time piano transcription visualiser.

## Thread Model

Shruti Parade uses a pipeline of 4–5 dedicated threads communicating via lock-free channels:

```
         ┌─────────────────┐
         │  Main Thread     │
         │  (winit event    │
         │   loop + wgpu    │
         │   rendering)     │
         └────────▲─────────┘
                  │ NoteEvent (crossbeam bounded channel)
         ┌────────┴─────────┐
         │  MIDI Router     │  T3
         │  (fan-out)       │
         └────────▲─────────┘
                  │ NoteEvent
         ┌────────┴─────────┐
         │  Inference       │  T2
         │  (onset/offset   │
         │   detection)     │
         └────────▲─────────┘
                  │ PitchFrame (crossbeam bounded channel)
         ┌────────┴─────────┐
         │  DSP             │  T1
         │  (FFT + harmonic │
         │   sieve)         │
         └────────▲─────────┘
                  │ f32 samples (rtrb SPSC ring buffer)
         ┌────────┴─────────┐
         │  Audio Ingest    │  T0
         │  (cpal callback  │
         │   or file decode)│
         └──────────────────┘
```

For MIDI file input, the pipeline is simplified: T0 parses the MIDI file, generates `NoteEvent`s directly, and drives a sine synthesiser for audio playback. Threads T1–T2 are not used.

## Data Flow

### 1. Audio Ingestion (`audio.rs`, `audio_file.rs`)

Audio samples enter the system as mono `f32` values:
- **Live capture**: cpal audio callback pushes samples into an `rtrb` ring buffer. The callback is real-time safe (no allocations, no locks, no syscalls).
- **Audio file**: symphonia decodes the file to PCM, then streams chunks at real-time pace using `thread::sleep` for pacing. Samples are pushed to both a DSP ring and a playback ring.

### 2. DSP Pipeline (`dsp.rs`)

The DSP thread consumes samples from the ring buffer and produces `PitchFrame`s:

```
Samples → CircBuf → Hann Window → Real FFT → Magnitudes → Harmonic Sieve → PitchFrame
```

#### Circular Buffer
A fixed-size circular buffer (`CircBuf`) accumulates incoming samples. On each hop, the buffer is linearised into a contiguous array for windowing. The overlap between frames is `fft_size - hop_size` samples.

#### Hann Window
Standard Hann window: `w(n) = 0.5 × (1 - cos(2πn/N))`. Applied element-wise to the linearised buffer. This reduces spectral leakage at the cost of slightly broadening spectral peaks.

#### Real FFT
Uses `realfft` (backed by `rustfft`) for an in-place forward FFT. Output is `N/2 + 1` complex bins representing frequencies from 0 to Nyquist.

#### Magnitude Spectrum
Complex FFT output → magnitude: `|X(k)| = √(Re² + Im²)`. The magnitudes are unnormalised (scale with FFT size), but all comparisons are relative.

#### Harmonic Sieve (`build_harmonic_templates`, `compute_pitch_energies`)

For each of the 88 piano keys (MIDI 21–108), a **harmonic template** pre-computes the FFT bin indices and weights for the fundamental and up to 8 harmonics:

- **Bin mapping**: `bin(h) = round(f₀ × h × N / SR)`, clamped to Nyquist × 0.95
- **Weights**: `w(h) = 1/h` — natural harmonic decay weighting
- **Neighbourhood**: ±1 bin around each harmonic is checked (max of 3 bins) to handle spectral leakage

The **pitch energy score** is the weighted arithmetic mean:

```
score(p) = Σ(w_h × max_mag(bin_h ± 1)) / Σ(w_h)
```

A **fundamental gate** rejects pitches whose fundamental magnitude is less than 10% of their score. This catches "ghost notes" where energy at upper harmonics coincidentally aligns with another pitch's template.

#### Harmonic Aliasing Suppression (`suppress_harmonic_aliasing`)

After scoring, pitches are processed from strongest to weakest:
1. **Upward harmonics**: Suppress weaker pitches at 2×, 3×, …, 8× the fundamental frequency (±1 semitone)
2. **Sub-harmonics**: Suppress weaker pitches at f₀/2, f₀/3, …, f₀/8 (±1 semitone)

A pitch is only suppressed if its energy is below `ratio_threshold` (default 50%) of the stronger pitch. This preserves real concurrent notes while eliminating harmonic coincidence artifacts.

### 3. Inference Engine (`inference.rs`)

The inference engine operates on `PitchFrame`s from the DSP thread:

1. **Local-max filter**: Only pitches that are local maxima within ±2 semitones survive
2. **Noise floor tracking**: Per-pitch adaptive noise floor with asymmetric rise/fall rates
3. **Onset detection**: Requires BOTH energy above SNR threshold AND frame-over-frame flux ratio > 1.5, confirmed across 2 consecutive frames
4. **Offset detection**: Triggered when energy falls below the noise floor or drops to 1/5 of peak energy for 8 consecutive frames
5. **Polyphony limit**: Maximum 10 simultaneous notes to prevent noise from flooding the output
6. **Retrigger cooldown**: 6-frame cooldown after note-off before the same pitch can re-onset

### 4. MIDI Router (`midi_router.rs`)

Simple fan-out: receives `NoteEvent`s and forwards them to the render thread. Uses `try_send` to drop events under back-pressure (visual fidelity is non-critical). Designed for future extension to MIDI output ports and SMF recording.

### 5. Renderer (`render.rs`)

The renderer runs on the main thread using winit's `ApplicationHandler` and wgpu:

#### Pipelines
- **Note pipeline**: Instanced SDF rounded-rectangles with per-note glow
- **Key pipeline**: 88 piano keys (52 white + 36 black) with active-note highlighting
- **Particle pipeline**: Additive-blended circular particles at note impact points
- **Hit-line pipeline**: Fullscreen pass drawing a glowing horizontal line

#### Bloom Post-Processing
Four fullscreen passes when `bloom_enabled`:
1. **Threshold**: Extract bright regions (luminance > 0.4)
2. **Horizontal blur**: 9-tap Gaussian, manually unrolled
3. **Vertical blur**: 9-tap Gaussian, manually unrolled
4. **Composite**: Additive blend of bloom with original scene

#### Visual Note Lifecycle
1. `NoteOn` → create `VisualNote` with `start_time`, `end_time = None`
2. While held: note rectangle extends downward from the hit-line
3. `NoteOff` → set `end_time`, rectangle becomes fixed-height
4. Notes scroll upward at `SCROLL_SPEED` (0.5 NDC/s) and are culled after 12 seconds

## Timing

The `AudioClock` (in `timing.rs`) is the single source of truth for wall-clock time. It's an atomic `u64` sample counter advanced by the audio callback thread:

- **Writer**: `advance(frames)` — `fetch_add` with `Release` ordering
- **Readers**: `now_seconds()` — `load` with `Acquire` ordering
- **Wait-free**: No mutex, no CAS loop

All timestamps in the system are derived from this clock, ensuring consistent synchronisation between audio, DSP, inference, and rendering.

## Frequency Resolution Considerations

At 48 kHz sample rate with 4096-point FFT:

| Register | Fundamental | Bin Resolution | Bins per Note |
|----------|-------------|----------------|---------------|
| A0 (27.5 Hz) | ~2.4 bins | 11.7 Hz | ~1.6 |
| C4 (261.6 Hz) | ~22.4 bins | 11.7 Hz | ~15.0 |
| A4 (440.0 Hz) | ~37.6 bins | 11.7 Hz | ~25.7 |
| C8 (4186 Hz) | ~357.8 bins | 11.7 Hz | ~220 |

Low-register notes (A0–C2) have poor fundamental resolution but benefit from the harmonic sieve, which aggregates energy across multiple well-resolved upper harmonics. Increasing `fft_size` to 8192 doubles resolution at the cost of ~85 ms latency per frame (vs ~43 ms at 4096).

## Dependencies

| Crate | Purpose |
|-------|---------|
| `cpal` | Cross-platform audio I/O |
| `rtrb` | Lock-free SPSC ring buffer |
| `crossbeam-channel` | Bounded MPSC channels |
| `rustfft` / `realfft` | FFT computation |
| `symphonia` | Audio file decoding (WAV/MP3/FLAC/OGG) |
| `midly` | MIDI file parsing |
| `wgpu` | GPU-accelerated rendering |
| `winit` | Windowing and event loop |
| `bytemuck` | Safe Pod casting for GPU buffers |
| `hound` | WAV file I/O (legacy, used by `wav_ingest.rs`) |
| `anyhow` / `thiserror` | Error handling |
| `tracing` | Structured logging |
| `serde` / `toml` | Configuration (de)serialisation |
