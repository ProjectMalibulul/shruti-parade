use crossbeam_channel::Sender;
use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;
use tracing::{debug, info};

use crate::config::DspConfig;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const PIANO_LO: u8 = 21; // A0
pub const PIANO_HI: u8 = 108; // C8

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Per-pitch energy frame produced by the harmonic-sieve DSP pipeline.
/// Each element is the harmonic-weighted energy for one MIDI pitch.
#[derive(Clone)]
pub struct PitchFrame {
    pub pitch_energy: [f32; 128], // indexed by MIDI pitch (0-127)
    pub sample_offset: u64,
}

// ---------------------------------------------------------------------------
// Circular buffer (O(1) push, O(n) linearise — amortised over hop)
// ---------------------------------------------------------------------------

pub struct CircBuf {
    buf: Vec<f32>,
    pos: usize,
}

impl CircBuf {
    pub fn new(size: usize) -> Self {
        Self {
            buf: vec![0.0; size],
            pos: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, sample: f32) {
        self.buf[self.pos] = sample;
        self.pos = (self.pos + 1) % self.buf.len();
    }

    /// Linearise oldest→newest into `out` (must be same length as internal buf).
    pub fn linearise_into(&self, out: &mut [f32]) {
        let n = self.buf.len();
        debug_assert_eq!(out.len(), n);
        let tail = &self.buf[self.pos..];
        let head = &self.buf[..self.pos];
        out[..tail.len()].copy_from_slice(tail);
        out[tail.len()..].copy_from_slice(head);
    }
}

// ---------------------------------------------------------------------------
// Harmonic template — pre-computed FFT bins + weights for one MIDI pitch
// ---------------------------------------------------------------------------

/// For each MIDI pitch we precompute which FFT bins correspond to its
/// fundamental and first several harmonics, plus a 1/h weighting.
pub struct HarmonicTemplate {
    /// FFT bin indices for each harmonic (h = 1, 2, 3, …)
    pub bins: Vec<usize>,
    /// Weight for each harmonic (1/h decay)
    pub weights: Vec<f32>,
}

/// Maximum number of harmonics to consider per pitch.
const MAX_HARMONICS: usize = 8;

/// Minimum ratio of the fundamental's magnitude to the overall weighted-mean
/// score.  Pitches whose fundamental is weaker than this fraction of their
/// score are likely false positives from harmonic coincidence with another
/// note's upper partials (e.g. perfect-fifth ghost notes).
const FUNDAMENTAL_GATE: f32 = 0.10;

/// Build harmonic templates for all 88 piano keys (MIDI 21–108).
/// Returns a 128-element Vec indexed by MIDI pitch; entries outside
/// the piano range are `None`.
pub fn build_harmonic_templates(
    sample_rate: f32,
    fft_size: usize,
) -> Vec<Option<HarmonicTemplate>> {
    let nyquist = sample_rate / 2.0;
    let n_bins = fft_size / 2 + 1;

    let mut templates: Vec<Option<HarmonicTemplate>> = (0..128).map(|_| None).collect();

    for midi in PIANO_LO..=PIANO_HI {
        let f0 = midi_to_hz(midi);
        let mut bins = Vec::new();
        let mut weights = Vec::new();

        for h in 1..=MAX_HARMONICS {
            let fh = f0 * h as f32;
            if fh >= nyquist * 0.95 {
                break;
            }

            let bin = (fh * fft_size as f32 / sample_rate).round() as usize;
            if bin >= n_bins {
                break;
            }

            bins.push(bin);
            weights.push(1.0 / h as f32);
        }

        if !bins.is_empty() {
            templates[midi as usize] = Some(HarmonicTemplate { bins, weights });
        }
    }

    templates
}

/// Compute per-pitch energy from FFT magnitudes using weighted harmonic sum.
///
/// Uses a **weighted arithmetic mean** of magnitudes at harmonic positions
/// (w_h = 1/h).  This is robust to missing upper harmonics — a real piano
/// note scores high even if harmonics 5–8 are weak or absent, while a
/// false-positive pitch (whose score comes from one harmonic coincidence
/// with another note) still scores low because most of its template bins
/// are empty.
///
/// A **fundamental gate** rejects pitches whose fundamental bin has very
/// little energy relative to the score.  This catches "ghost notes" caused
/// by non-integer harmonic coincidence (e.g. a perfect fifth above/below a
/// strong note).
///
/// `score = Σ(w_h · mag_h) / Σ(w_h)`  where w_h = 1/h
pub fn compute_pitch_energies(
    magnitudes: &[f32],
    templates: &[Option<HarmonicTemplate>],
) -> [f32; 128] {
    let mut energies = [0.0f32; 128];
    let n_bins = magnitudes.len();

    for midi in PIANO_LO..=PIANO_HI {
        if let Some(ref tmpl) = templates[midi as usize] {
            let mut weighted_sum = 0.0f32;
            let mut total_weight = 0.0f32;
            let mut fund_mag = 0.0f32;

            for (idx, (&bin, &weight)) in tmpl.bins.iter().zip(tmpl.weights.iter()).enumerate() {
                let lo = bin.saturating_sub(1);
                let hi = (bin + 1).min(n_bins - 1);
                let mag = magnitudes[lo..=hi].iter().copied().fold(0.0f32, f32::max);

                if idx == 0 {
                    fund_mag = mag;
                }

                weighted_sum += weight * mag;
                total_weight += weight;
            }

            if total_weight > 0.0 {
                let score = weighted_sum / total_weight;
                // Fundamental gate: reject pitches whose fundamental is too
                // weak relative to the score (harmonic-coincidence ghosts).
                if fund_mag >= score * FUNDAMENTAL_GATE {
                    energies[midi as usize] = score;
                }
            }
        }
    }

    energies
}

/// Suppress pitches whose energy is explained by harmonics of a stronger pitch.
///
/// Process from strongest pitch to weakest. For each fundamental:
///   1. **Upward harmonics**: suppress pitches at h × f0 (h = 2..8)
///   2. **Sub-harmonics**: suppress pitches at f0 / h (h = 2..8) that
///      have a higher harmonic landing on f0.
///
/// In both directions, ±1 semitone around the target is checked to
/// account for spectral leakage from ±1 FFT bin neighbourhood.
///
/// Pitches with energy ≥ `ratio_threshold` of the stronger pitch are
/// kept — they may be real concurrent notes.
pub fn suppress_harmonic_aliasing(energies: &mut [f32; 128], ratio_threshold: f32) {
    // Collect active pitches sorted by energy (strongest first)
    let mut sorted: Vec<(usize, f32)> = (PIANO_LO as usize..=PIANO_HI as usize)
        .filter(|&p| energies[p] > 0.0)
        .map(|p| (p, energies[p]))
        .collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut suppressed = [false; 128];

    for &(fund_midi, fund_energy) in &sorted {
        if suppressed[fund_midi] {
            continue;
        }

        let f0 = midi_to_hz(fund_midi as u8);

        // ---- Upward harmonics: pitches at h × f0 ----
        for h in 2..=MAX_HARMONICS {
            let fh = f0 * h as f32;
            let midi_f = 69.0 + 12.0 * (fh / 440.0).log2();
            let midi_round = midi_f.round() as i32;

            // Check ±1 semitone to catch spectral leakage
            for offset in -1..=1i32 {
                let check = midi_round + offset;
                if check >= PIANO_LO as i32 && check <= PIANO_HI as i32 {
                    let ci = check as usize;
                    if !suppressed[ci]
                        && ci != fund_midi
                        && energies[ci] < fund_energy * ratio_threshold
                    {
                        energies[ci] = 0.0;
                        suppressed[ci] = true;
                    }
                }
            }
        }

        // ---- Sub-harmonics: pitches at f0 / h ----
        // These are pitches whose h-th harmonic coincides with our fundamental.
        for h in 2..=MAX_HARMONICS {
            let fh = f0 / h as f32;
            if fh < 20.0 {
                continue;
            }
            let midi_f = 69.0 + 12.0 * (fh / 440.0).log2();
            let midi_round = midi_f.round() as i32;

            for offset in -1..=1i32 {
                let check = midi_round + offset;
                if check >= PIANO_LO as i32 && check <= PIANO_HI as i32 {
                    let ci = check as usize;
                    if !suppressed[ci]
                        && ci != fund_midi
                        && energies[ci] < fund_energy * ratio_threshold
                    {
                        energies[ci] = 0.0;
                        suppressed[ci] = true;
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DSP pipeline
// ---------------------------------------------------------------------------

pub struct DspPipeline {
    config: DspConfig,
    consumer: rtrb::Consumer<f32>,
    pitch_tx: Sender<PitchFrame>,
    circ: CircBuf,
    hann: Vec<f32>,
    fft_in: Vec<f32>,
    fft_out: Vec<Complex<f32>>,
    magnitudes: Vec<f32>,
    harmonic_templates: Vec<Option<HarmonicTemplate>>,
    sample_count: u64,
}

impl DspPipeline {
    pub fn new(
        config: DspConfig,
        consumer: rtrb::Consumer<f32>,
        pitch_tx: Sender<PitchFrame>,
        sample_rate: u32,
    ) -> Self {
        let n_bins = config.fft_size / 2 + 1;

        let hann: Vec<f32> = (0..config.fft_size)
            .map(|i| {
                let phase = 2.0 * std::f32::consts::PI * i as f32 / config.fft_size as f32;
                0.5 * (1.0 - phase.cos())
            })
            .collect();

        let harmonic_templates = build_harmonic_templates(sample_rate as f32, config.fft_size);

        Self {
            circ: CircBuf::new(config.fft_size),
            hann,
            fft_in: vec![0.0; config.fft_size],
            fft_out: vec![Complex::default(); n_bins],
            magnitudes: vec![0.0; n_bins],
            harmonic_templates,
            consumer,
            pitch_tx,
            sample_count: 0,
            config,
        }
    }

    /// Blocking DSP loop — call from a dedicated thread.
    pub fn run(&mut self) {
        info!(
            "DSP pipeline started (fft={}, hop={}, harmonic sieve)",
            self.config.fft_size, self.config.hop_size,
        );

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.config.fft_size);
        let mut scratch = fft.make_scratch_vec();

        loop {
            // ---- collect hop_size new samples ----
            let hop = self.config.hop_size;
            let mut collected = 0usize;

            while collected < hop {
                match self.consumer.pop() {
                    Ok(s) => {
                        self.circ.push(s);
                        collected += 1;
                    }
                    Err(_) => {
                        // Ring empty — brief back-off (well under hop period).
                        std::thread::sleep(std::time::Duration::from_micros(200));
                    }
                }
            }

            self.sample_count += hop as u64;

            // ---- linearise circular buffer + apply Hann window ----
            self.circ.linearise_into(&mut self.fft_in);
            for (s, w) in self.fft_in.iter_mut().zip(self.hann.iter()) {
                *s *= w;
            }

            // ---- real FFT ----
            if fft
                .process_with_scratch(&mut self.fft_in, &mut self.fft_out, &mut scratch)
                .is_err()
            {
                continue;
            }

            // ---- compute magnitudes ----
            for (i, c) in self.fft_out.iter().enumerate() {
                self.magnitudes[i] = (c.re * c.re + c.im * c.im).sqrt();
            }

            // ---- harmonic sieve → per-pitch energies ----
            let pitch_energy = compute_pitch_energies(&self.magnitudes, &self.harmonic_templates);

            if self
                .pitch_tx
                .send(PitchFrame {
                    pitch_energy,
                    sample_offset: self.sample_count,
                })
                .is_err()
            {
                debug!("Pitch channel closed — DSP shutting down");
                return;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Utility: MIDI pitch → frequency
// ---------------------------------------------------------------------------

pub fn midi_to_hz(midi: u8) -> f32 {
    440.0 * 2.0f32.powf((midi as f32 - 69.0) / 12.0)
}

// ---------------------------------------------------------------------------
// Mel-filterbank construction (kept for tests / future use)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

#[allow(dead_code)]
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

#[allow(dead_code)]
pub fn build_mel_filterbank(
    n_mels: usize,
    fft_size: usize,
    sr: f32,
    fmin: f32,
    fmax: f32,
) -> Vec<Vec<f32>> {
    let n_bins = fft_size / 2 + 1;
    let mel_lo = hz_to_mel(fmin);
    let mel_hi = hz_to_mel(fmax);

    let pts: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_lo + (mel_hi - mel_lo) * i as f32 / (n_mels + 1) as f32)
        .collect();
    let bins: Vec<f32> = pts
        .iter()
        .map(|&m| mel_to_hz(m) * fft_size as f32 / sr)
        .collect();

    (0..n_mels)
        .map(|m| {
            let (l, c, r) = (bins[m], bins[m + 1], bins[m + 2]);
            (0..n_bins)
                .map(|k| {
                    let kf = k as f32;
                    if kf >= l && kf <= c {
                        (kf - l) / (c - l + 1e-10)
                    } else if kf > c && kf <= r {
                        (r - kf) / (r - c + 1e-10)
                    } else {
                        0.0
                    }
                })
                .collect()
        })
        .collect()
}
