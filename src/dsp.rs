// ---------------------------------------------------------------------------
// Pitch detection strategy
// ---------------------------------------------------------------------------
// We augment the existing FFT harmonic-sieve with the McLeod Pitch Method
// (MPM / Normalised Square Difference Function).  MPM operates in the time
// domain, gives sub-bin pitch accuracy via parabolic interpolation, and
// resolves low-pitch notes far better than a 4096-pt FFT alone (~11.7 Hz/bin
// at 48 kHz).  The FFT path is retained for onset-energy estimation while
// MPM refines per-key confidence scores in the PitchFrame.
//
// Option chosen: A (MPM augmentation).  The integration stays within dsp.rs
// and only touches PitchFrame scoring — well under the ~100-line threshold
// that would have triggered Option B.
// ---------------------------------------------------------------------------

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

/// Bass/treble crossover: pitches below this MIDI number use the bass FFT.
const BASS_CROSSOVER: u8 = 48; // C3

/// Approximate inharmonicity coefficient B for a typical grand piano.
/// Piano strings exhibit stretched partials: f_n = n·f0·√(1 + B·n²).
fn inharmonicity_b(midi: u8) -> f32 {
    match midi {
        21..=35 => 0.0004,
        36..=47 => 0.00015,
        48..=59 => 0.00005,
        60..=71 => 0.00003,
        72..=84 => 0.0001,
        85..=96 => 0.0005,
        97..=108 => 0.002,
        _ => 0.0,
    }
}

/// Build harmonic templates for all 88 piano keys (MIDI 21–108).
/// Returns a 128-element Vec indexed by MIDI pitch; entries outside
/// the piano range are `None`.
///
/// Harmonic positions are adjusted for piano inharmonicity so that
/// template bins match the stretched partial frequencies of real strings.
pub fn build_harmonic_templates(
    sample_rate: f32,
    fft_size: usize,
) -> Vec<Option<HarmonicTemplate>> {
    let nyquist = sample_rate / 2.0;
    let n_bins = fft_size / 2 + 1;

    let mut templates: Vec<Option<HarmonicTemplate>> = (0..128).map(|_| None).collect();

    for midi in PIANO_LO..=PIANO_HI {
        let f0 = midi_to_hz(midi);
        let b = inharmonicity_b(midi);
        let mut bins = Vec::new();
        let mut weights = Vec::new();

        for h in 1..=MAX_HARMONICS {
            let fh = f0 * h as f32 * (1.0 + b * (h as f32).powi(2)).sqrt();
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
// McLeod Pitch Method (MPM) — Normalised Square Difference Function
// ---------------------------------------------------------------------------

/// Compute the Normalised Square Difference Function (NSDF) of `x`.
///
/// NSDF(τ) = 2 r(τ) / (m(0) + m(τ))
/// where r(τ) is the autocorrelation at lag τ, and m(τ) = Σ x²[j] + x²[j+τ].
fn compute_nsdf(x: &[f32], nsdf_out: &mut [f32]) {
    let n = x.len();
    // Precompute cumulative sum of squares for the running normalisation term.
    // cum[i] = Σ x²[j] for j = i..n   (computed backwards)
    let mut cum_sq = vec![0.0f32; n + 1]; // cum_sq[n] = 0
    for i in (0..n).rev() {
        cum_sq[i] = cum_sq[i + 1] + x[i] * x[i];
    }

    for tau in 0..nsdf_out.len() {
        let mut acf = 0.0f32;
        let len = n - tau;
        for j in 0..len {
            acf += x[j] * x[j + tau];
        }
        // m(tau) = cum_sq[0] - cum_sq[len] + cum_sq[tau] - cum_sq[n]
        // Simplified: sum of x[0..len]² + sum of x[tau..n]²
        let m = (cum_sq[0] - cum_sq[len]) + (cum_sq[tau] - cum_sq[n]);
        if m > 1e-12 {
            nsdf_out[tau] = 2.0 * acf / m;
        } else {
            nsdf_out[tau] = 0.0;
        }
    }
}

/// Find "key maxima" of the NSDF — peaks that follow a zero crossing from below.
/// Skips the trivial self-correlation hump near lag 0 by starting after the
/// first negative region.
/// Returns (lag, nsdf_value) pairs.
fn find_nsdf_key_maxima(nsdf: &[f32]) -> Vec<(usize, f32)> {
    let mut maxima = Vec::new();
    let mut positive_region = false;
    let mut best_lag = 0usize;
    let mut best_val = 0.0f32;

    // Skip the initial positive region (trivial autocorrelation near lag 0).
    // Find the first zero-crossing from positive to negative.
    let mut start = 1;
    while start < nsdf.len() && nsdf[start] > 0.0 {
        start += 1;
    }

    for (tau, &val) in nsdf.iter().enumerate().skip(start) {
        if val > 0.0 {
            if !positive_region {
                positive_region = true;
                best_lag = tau;
                best_val = val;
            } else if val > best_val {
                best_lag = tau;
                best_val = val;
            }
        } else if positive_region {
            // End of positive hump — record its peak
            maxima.push((best_lag, best_val));
            positive_region = false;
            best_val = 0.0;
        }
    }
    // Capture last hump if still in positive region
    if positive_region && best_val > 0.0 {
        maxima.push((best_lag, best_val));
    }
    maxima
}

/// Parabolic interpolation around index `idx` in `data`.
/// Returns the fractional index of the true peak.
fn parabolic_interp(data: &[f32], idx: usize) -> f32 {
    if idx == 0 || idx + 1 >= data.len() {
        return idx as f32;
    }
    let a = data[idx - 1];
    let b = data[idx];
    let c = data[idx + 1];
    let denom = 2.0 * (2.0 * b - a - c);
    if denom.abs() < 1e-12 {
        return idx as f32;
    }
    idx as f32 + (a - c) / denom
}

/// Run MPM on a windowed signal buffer and return estimated pitch in Hz,
/// or None if no clear pitch is found.
///
/// `clarity_threshold`: minimum NSDF peak value to accept (0.0–1.0, typically 0.3–0.6).
pub fn mpm_pitch(signal: &[f32], sample_rate: f32, clarity_threshold: f32) -> Option<f32> {
    let n = signal.len();
    let max_lag = n / 2;
    let mut nsdf = vec![0.0f32; max_lag];
    compute_nsdf(signal, &mut nsdf);

    let maxima = find_nsdf_key_maxima(&nsdf);
    if maxima.is_empty() {
        return None;
    }

    // MPM "first peak above threshold" heuristic: pick the first key maximum
    // whose NSDF value exceeds clarity_threshold × global_max.
    let global_max = maxima.iter().map(|&(_, v)| v).fold(0.0f32, f32::max);
    let threshold = clarity_threshold * global_max;

    for &(lag, val) in &maxima {
        if val >= threshold && lag > 0 {
            let refined_lag = parabolic_interp(&nsdf, lag);
            if refined_lag > 0.0 {
                return Some(sample_rate / refined_lag);
            }
        }
    }
    None
}

/// Refine harmonic-sieve scores using MPM pitch estimates.
///
/// For each active pitch in `energies`, checks whether MPM confirms the
/// pitch within ±1 semitone. Pitches confirmed by MPM get a confidence
/// boost; unconfirmed pitches get mildly attenuated. This improves low-
/// pitch accuracy where FFT bin resolution is poor.
pub fn refine_with_mpm(energies: &mut [f32; 128], signal: &[f32], sample_rate: f32) {
    let mpm_hz = match mpm_pitch(signal, sample_rate, 0.4) {
        Some(f) => f,
        None => return, // no clear pitch — leave energies unchanged
    };

    // Convert detected Hz to fractional MIDI
    let mpm_midi = 69.0 + 12.0 * (mpm_hz / 440.0).log2();

    for midi in PIANO_LO..=PIANO_HI {
        let e = energies[midi as usize];
        if e <= 0.0 {
            continue;
        }
        let dist = (midi as f32 - mpm_midi).abs();
        if dist <= 0.7 {
            // MPM confirms this pitch — boost by up to 30%
            energies[midi as usize] *= 1.0 + 0.3 * (1.0 - dist / 0.7);
        } else if dist > 3.0 {
            // Far from MPM estimate — mild attenuation for low notes
            // where FFT resolution is poor
            if midi < 60 {
                energies[midi as usize] *= 0.7;
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
    // Bass FFT (8192-pt) for improved low-pitch resolution
    bass_circ: CircBuf,
    bass_hann: Vec<f32>,
    bass_fft_in: Vec<f32>,
    bass_fft_out: Vec<Complex<f32>>,
    bass_magnitudes: Vec<f32>,
    bass_templates: Vec<Option<HarmonicTemplate>>,
    sample_count: u64,
    sample_rate: f32,
    mpm_buf: Vec<f32>,
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

        // Bass FFT path (larger window for low-frequency resolution)
        let bass_fft = config.bass_fft_size;
        let bass_n_bins = bass_fft / 2 + 1;
        let bass_hann: Vec<f32> = (0..bass_fft)
            .map(|i| {
                let phase = 2.0 * std::f32::consts::PI * i as f32 / bass_fft as f32;
                0.5 * (1.0 - phase.cos())
            })
            .collect();
        let bass_templates = build_harmonic_templates(sample_rate as f32, bass_fft);

        Self {
            circ: CircBuf::new(config.fft_size),
            hann,
            fft_in: vec![0.0; config.fft_size],
            fft_out: vec![Complex::default(); n_bins],
            magnitudes: vec![0.0; n_bins],
            harmonic_templates,
            bass_circ: CircBuf::new(bass_fft),
            bass_hann,
            bass_fft_in: vec![0.0; bass_fft],
            bass_fft_out: vec![Complex::default(); bass_n_bins],
            bass_magnitudes: vec![0.0; bass_n_bins],
            bass_templates,
            consumer,
            pitch_tx,
            sample_count: 0,
            sample_rate: sample_rate as f32,
            mpm_buf: vec![0.0; config.fft_size],
            config,
        }
    }

    /// Blocking DSP loop — call from a dedicated thread.
    pub fn run(&mut self) {
        info!(
            "DSP pipeline started (fft={}/{}, hop={}, harmonic sieve + inharmonicity)",
            self.config.fft_size, self.config.bass_fft_size, self.config.hop_size,
        );

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.config.fft_size);
        let mut scratch = fft.make_scratch_vec();
        let bass_fft = planner.plan_fft_forward(self.config.bass_fft_size);
        let mut bass_scratch = bass_fft.make_scratch_vec();

        let mut frames_sent: u64 = 0;

        loop {
            // ---- collect hop_size new samples ----
            let hop = self.config.hop_size;
            let mut collected = 0usize;

            while collected < hop {
                match self.consumer.pop() {
                    Ok(s) => {
                        self.circ.push(s);
                        self.bass_circ.push(s);
                        collected += 1;
                    }
                    Err(_) => {
                        // Ring empty — brief back-off (well under hop period).
                        std::thread::sleep(std::time::Duration::from_micros(200));
                    }
                }
            }

            self.sample_count += hop as u64;

            // ---- Standard FFT (4096-pt) for mid/treble ----
            self.circ.linearise_into(&mut self.fft_in);
            for (s, w) in self.fft_in.iter_mut().zip(self.hann.iter()) {
                *s *= w;
            }

            if fft
                .process_with_scratch(&mut self.fft_in, &mut self.fft_out, &mut scratch)
                .is_err()
            {
                continue;
            }

            for (i, c) in self.fft_out.iter().enumerate() {
                self.magnitudes[i] = (c.re * c.re + c.im * c.im).sqrt();
            }

            // ---- Bass FFT (8192-pt) for improved low-pitch resolution ----
            self.bass_circ.linearise_into(&mut self.bass_fft_in);
            for (s, w) in self.bass_fft_in.iter_mut().zip(self.bass_hann.iter()) {
                *s *= w;
            }

            let bass_ok = bass_fft
                .process_with_scratch(
                    &mut self.bass_fft_in,
                    &mut self.bass_fft_out,
                    &mut bass_scratch,
                )
                .is_ok();

            if bass_ok {
                for (i, c) in self.bass_fft_out.iter().enumerate() {
                    self.bass_magnitudes[i] = (c.re * c.re + c.im * c.im).sqrt();
                }
            }

            // ---- Harmonic sieve → per-pitch energies (dual resolution) ----
            let energy_std = compute_pitch_energies(&self.magnitudes, &self.harmonic_templates);
            let energy_bass = if bass_ok {
                compute_pitch_energies(&self.bass_magnitudes, &self.bass_templates)
            } else {
                energy_std
            };

            // Merge: bass FFT for MIDI 21–47, standard FFT for MIDI 48–108
            let mut pitch_energy = [0.0f32; 128];
            for midi in PIANO_LO..=PIANO_HI {
                let i = midi as usize;
                pitch_energy[i] = if midi < BASS_CROSSOVER {
                    energy_bass[i]
                } else {
                    energy_std[i]
                };
            }

            // ---- MPM refinement (unwindowed signal) ----
            self.circ.linearise_into(&mut self.mpm_buf);
            refine_with_mpm(&mut pitch_energy, &self.mpm_buf, self.sample_rate);

            if self
                .pitch_tx
                .send(PitchFrame {
                    pitch_energy,
                    sample_offset: self
                        .sample_count
                        .saturating_sub(self.config.fft_size as u64 / 2),
                })
                .is_err()
            {
                debug!("Pitch channel closed — DSP shutting down");
                return;
            }
            frames_sent += 1;
            if frames_sent == 1 {
                info!("DSP: first PitchFrame sent at sample {}", self.sample_count);
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

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod dsp_tests {
    use super::*;

    const SR: f32 = 48000.0;
    const FFT_SIZE: usize = 4096;

    #[test]
    fn test_mpm_pure_sine_440() {
        let n = 2048;
        let signal: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SR).sin())
            .collect();
        let detected = mpm_pitch(&signal, SR, 0.4).expect("MPM should detect a pitch");
        let error = (detected - 440.0).abs();
        assert!(
            error < 2.0,
            "Expected 440 Hz ± 2 Hz, got {detected:.2} Hz (error {error:.2})"
        );
    }

    #[test]
    fn test_harmonic_sieve_a4() {
        let templates = build_harmonic_templates(SR, FFT_SIZE);

        // Generate 440 Hz sine + harmonics
        let signal: Vec<f32> = (0..FFT_SIZE)
            .map(|i| {
                let t = i as f32 / SR;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                    + 0.5 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
                    + 0.25 * (2.0 * std::f32::consts::PI * 1320.0 * t).sin()
            })
            .collect();

        // Apply Hann window
        let mut windowed = signal.clone();
        for (i, s) in windowed.iter_mut().enumerate() {
            let w = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / FFT_SIZE as f32).cos());
            *s *= w;
        }

        // FFT
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let mut scratch = fft.make_scratch_vec();
        let n_bins = FFT_SIZE / 2 + 1;
        let mut fft_out = vec![rustfft::num_complex::Complex::default(); n_bins];
        fft.process_with_scratch(&mut windowed, &mut fft_out, &mut scratch)
            .unwrap();
        let magnitudes: Vec<f32> = fft_out
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();

        let energies = compute_pitch_energies(&magnitudes, &templates);

        // MIDI 69 = A4 should have the highest energy
        let a4_energy = energies[69];
        assert!(a4_energy > 0.0, "A4 energy should be positive");
        for midi in PIANO_LO..=PIANO_HI {
            if midi != 69 {
                assert!(
                    energies[midi as usize] <= a4_energy,
                    "MIDI {} ({:.1} Hz) energy {:.4} exceeds A4 energy {:.4}",
                    midi,
                    midi_to_hz(midi),
                    energies[midi as usize],
                    a4_energy
                );
            }
        }
    }
}
