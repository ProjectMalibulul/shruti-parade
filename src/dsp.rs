use crossbeam_channel::Sender;
use realfft::RealFftPlanner;
use rustfft::num_complex::Complex;
use tracing::{debug, info};

use crate::config::DspConfig;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single mel-spectrogram frame emitted by the DSP pipeline.
#[derive(Clone)]
pub struct MelFrame {
    pub data: Vec<f32>,      // length = n_mels
    pub sample_offset: u64,  // sample index of frame centre
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
// DSP pipeline
// ---------------------------------------------------------------------------

pub struct DspPipeline {
    config: DspConfig,
    consumer: rtrb::Consumer<f32>,
    mel_tx: Sender<MelFrame>,
    circ: CircBuf,
    hann: Vec<f32>,
    fft_in: Vec<f32>,
    fft_out: Vec<Complex<f32>>,
    mel_fb: Vec<Vec<f32>>,
    sample_count: u64,
}

impl DspPipeline {
    pub fn new(
        config: DspConfig,
        consumer: rtrb::Consumer<f32>,
        mel_tx: Sender<MelFrame>,
        sample_rate: u32,
    ) -> Self {
        let n_bins = config.fft_size / 2 + 1;

        let hann: Vec<f32> = (0..config.fft_size)
            .map(|i| {
                let phase = 2.0 * std::f32::consts::PI * i as f32 / config.fft_size as f32;
                0.5 * (1.0 - phase.cos())
            })
            .collect();

        let mel_fb = build_mel_filterbank(
            config.n_mels,
            config.fft_size,
            sample_rate as f32,
            config.mel_fmin,
            config.mel_fmax,
        );

        Self {
            circ: CircBuf::new(config.fft_size),
            hann,
            fft_in: vec![0.0; config.fft_size],
            fft_out: vec![Complex::default(); n_bins],
            mel_fb,
            consumer,
            mel_tx,
            sample_count: 0,
            config,
        }
    }

    /// Blocking DSP loop — call from a dedicated thread.
    pub fn run(&mut self) {
        info!(
            "DSP pipeline started (fft={}, hop={}, mels={})",
            self.config.fft_size, self.config.hop_size, self.config.n_mels
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

            // ---- magnitude → mel filterbank → log ----
            let mel: Vec<f32> = self
                .mel_fb
                .iter()
                .map(|filt| {
                    let energy: f32 = filt
                        .iter()
                        .zip(self.fft_out.iter())
                        .map(|(&w, c)| w * (c.re * c.re + c.im * c.im).sqrt())
                        .sum();
                    (energy.max(1e-10)).ln()
                })
                .collect();

            if self
                .mel_tx
                .send(MelFrame {
                    data: mel,
                    sample_offset: self.sample_count,
                })
                .is_err()
            {
                debug!("Mel channel closed — DSP shutting down");
                return;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Mel-filterbank construction (HTK-style triangular filters)
// ---------------------------------------------------------------------------

pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

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
