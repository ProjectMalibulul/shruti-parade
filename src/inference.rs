use crossbeam_channel::{Receiver, Sender};
use tracing::{debug, info};

use crate::config::{DspConfig, InferenceConfig};
use crate::dsp::{self, MelFrame};
use crate::events::{NoteEvent, NoteEventKind};

const PIANO_LO: u8 = 21;  // A0
const PIANO_HI: u8 = 108; // C8
const N_PITCHES: usize = 128;

/// Maximum simultaneous notes — prevents noise from triggering the full keyboard.
const MAX_POLYPHONY: usize = 10;

/// Energy + flux onset detector v4 with:
///   - Correct flux handling for silence→note transitions
///   - Spectral-leakage suppression (adjacent semitone masking)
///   - Harmonic partial suppression
///   - Adaptive noise floor with warmup calibration
///   - Polyphony limiting
///   - Anti-chatter cooldown
pub struct InferenceEngine {
    config: InferenceConfig,
    dsp_config: DspConfig,
    mel_rx: Receiver<MelFrame>,
    event_tx: Sender<NoteEvent>,
}

impl InferenceEngine {
    pub fn new(
        config: InferenceConfig,
        dsp_config: DspConfig,
        mel_rx: Receiver<MelFrame>,
        event_tx: Sender<NoteEvent>,
    ) -> Self {
        Self {
            config,
            dsp_config,
            mel_rx,
            event_tx,
        }
    }

    /// Blocking onset-detection loop — call from a dedicated thread.
    pub fn run(&mut self) {
        info!("Inference engine started (onset detector v4)");

        let mut bin_to_pitch: Option<Vec<Option<u8>>> = None;

        // Per-pitch state arrays
        let mut prev_energy = [0.0f32; N_PITCHES];
        let mut noise_floor = [0.0f32; N_PITCHES];
        let mut active = [false; N_PITCHES];
        let mut onset_pending = [0u8; N_PITCHES];
        let mut pending_vel = [0u8; N_PITCHES];
        let mut peak_energy = [0.0f32; N_PITCHES];
        let mut below_count = [0u32; N_PITCHES];
        let mut cooldown = [0u32; N_PITCHES];
        let mut active_count: usize = 0;

        // Noise floor starts very low — warmup will calibrate it
        for nf in noise_floor.iter_mut() {
            *nf = 0.001;
        }

        // ---- Tuning constants ----
        // DSP outputs log-domain mel energies; we convert back with exp().
        // Measured ranges: silence ≈ 1e-10, noise ≈ 0.01, moderate note ≈ 50–400,
        // loud note ≈ 200–1000.
        let abs_min_energy: f32 = 0.5;
        // SNR: energy must exceed noise floor by this factor for onset.
        let snr_onset: f32 = 6.0 + self.config.onset_threshold * 8.0; // default: 10.0
        // Minimum frame-over-frame flux ratio — only checked on the FIRST
        // detection frame. When prev ≈ 0, we treat it as infinite flux.
        let flux_ratio: f32 = 1.8;
        // Noise floor adaptation
        let nf_rise: f32 = 0.003;
        let nf_fall: f32 = 0.06;
        // Onset confirmation: require N consecutive above-threshold frames
        let onset_confirm: u8 = 2;
        // Offset parameters
        let release_ratio: f32 = 2.0 + self.config.frame_threshold * 2.0; // default: 2.6
        let release_frames: u32 = 8;
        let peak_decay_ratio: f32 = 5.0;
        // Cooldown after NoteOff before the same pitch can retrigger
        let retrigger_cooldown: u32 = 6;

        let mut frame_num: u64 = 0;
        let mut warmup_done = false;

        for mel in self.mel_rx.iter() {
            let sample_time = mel.sample_offset;
            let n_mels = mel.data.len();
            frame_num += 1;

            // Lazily build bin→pitch map
            let btp = bin_to_pitch.get_or_insert_with(|| {
                build_bin_to_pitch(
                    n_mels,
                    self.dsp_config.mel_fmin,
                    self.dsp_config.mel_fmax,
                )
            });

            // ---- 1. Per-pitch energy: MAX of mel bins mapping to that pitch ----
            let mut pitch_energy = [0.0f32; N_PITCHES];

            for (bin, &log_e) in mel.data.iter().enumerate() {
                if let Some(&Some(midi)) = btp.get(bin) {
                    let linear = log_e.exp();
                    if linear > pitch_energy[midi as usize] {
                        pitch_energy[midi as usize] = linear;
                    }
                }
            }

            // ---- 2. Spectral leakage + harmonic suppression ----
            // For each pitch, if a neighbouring pitch (±1 semitone) or harmonic
            // interval pitch has >= 1.5× more energy, suppress this pitch.
            let raw_energy = pitch_energy;
            suppress_harmonics_and_leakage(&mut pitch_energy, &raw_energy);

            // ---- 3. Warmup: calibrate noise floor from first 30 frames ----
            if frame_num <= 30 {
                for i in PIANO_LO as usize..=PIANO_HI as usize {
                    noise_floor[i] += 0.2 * (pitch_energy[i] - noise_floor[i]);
                }
                for i in 0..N_PITCHES {
                    prev_energy[i] = pitch_energy[i];
                }
                continue;
            }

            if !warmup_done {
                warmup_done = true;
                for i in PIANO_LO as usize..=PIANO_HI as usize {
                    noise_floor[i] *= 1.5;
                    if noise_floor[i] < 0.005 {
                        noise_floor[i] = 0.005;
                    }
                }
                info!("Noise floor calibrated (30 frames)");
            }

            // ---- 4. Per-pitch onset / offset ----
            for pitch in PIANO_LO..=PIANO_HI {
                let i = pitch as usize;
                let e = pitch_energy[i];
                let prev = prev_energy[i];

                // Tick cooldown
                if cooldown[i] > 0 {
                    cooldown[i] -= 1;
                }

                // Update noise floor (only from inactive, non-pending pitches)
                if !active[i] && onset_pending[i] == 0 {
                    let alpha = if e > noise_floor[i] { nf_rise } else { nf_fall };
                    noise_floor[i] += alpha * (e - noise_floor[i]);
                    if noise_floor[i] < 0.01 {
                        noise_floor[i] = 0.01;
                    }
                }

                let onset_thresh = (noise_floor[i] * snr_onset).max(abs_min_energy);

                // Flux: current / previous ratio.
                // CRITICAL: when prev is near-zero and energy is above threshold,
                // that's a huge rise — treat as infinite flux, not zero.
                let flux = if prev < 1e-4 {
                    if e > abs_min_energy { f32::INFINITY } else { 0.0 }
                } else {
                    e / prev
                };

                if !active[i] {
                    if onset_pending[i] > 0 {
                        // Already pending: only need sustained energy (no flux re-check)
                        if e > onset_thresh && cooldown[i] == 0 {
                            onset_pending[i] += 1;
                            if onset_pending[i] >= onset_confirm {
                                // Polyphony check
                                if active_count >= MAX_POLYPHONY {
                                    onset_pending[i] = 0;
                                } else {
                                    active[i] = true;
                                    onset_pending[i] = 0;
                                    peak_energy[i] = e;
                                    below_count[i] = 0;
                                    active_count += 1;

                                    let _ = self.event_tx.send(NoteEvent {
                                        kind: NoteEventKind::NoteOn,
                                        pitch,
                                        velocity: pending_vel[i],
                                        sample_time,
                                    });
                                }
                            }
                        } else {
                            onset_pending[i] = 0;
                        }
                    } else {
                        // First detection: require BOTH energy threshold AND flux
                        if e > onset_thresh && flux > flux_ratio && cooldown[i] == 0 {
                            onset_pending[i] = 1;
                            let ratio = e / onset_thresh;
                            pending_vel[i] =
                                ((ratio.sqrt() * 20.0 + 40.0).clamp(40.0, 127.0)) as u8;
                        }
                    }
                } else {
                    // ---- Active note: track peak & detect offset ----
                    if e > peak_energy[i] {
                        peak_energy[i] = e;
                    }

                    let off_thresh =
                        (noise_floor[i] * release_ratio).max(abs_min_energy * 0.5);
                    let peak_drop = peak_energy[i] / peak_decay_ratio;

                    if e < off_thresh || e < peak_drop {
                        below_count[i] += 1;
                        if below_count[i] >= release_frames {
                            active[i] = false;
                            active_count = active_count.saturating_sub(1);
                            cooldown[i] = retrigger_cooldown;
                            let _ = self.event_tx.send(NoteEvent {
                                kind: NoteEventKind::NoteOff,
                                pitch,
                                velocity: 0,
                                sample_time,
                            });
                        }
                    } else {
                        below_count[i] = 0;
                    }
                }

                prev_energy[i] = e;
            }
        }

        // Emit NoteOff for still-active notes at shutdown
        for pitch in PIANO_LO..=PIANO_HI {
            if active[pitch as usize] {
                let _ = self.event_tx.send(NoteEvent {
                    kind: NoteEventKind::NoteOff,
                    pitch,
                    velocity: 0,
                    sample_time: 0,
                });
            }
        }

        debug!("Inference engine stopped (channel closed)");
    }
}

// ---------------------------------------------------------------------------
// Harmonic + spectral-leakage suppression
// ---------------------------------------------------------------------------

/// Three-stage suppression:
///   1. **Absolute floor**: Zero out anything below a minimum energy.
///   2. **Local-maxima filter**: Zero out any pitch that isn't a local energy peak
///      within ±2 semitones — eliminates spectral leakage between adjacent mel bins.
///   3. **Harmonic masking**: For each surviving pitch (strongest first),
///      **unconditionally** suppress pitches at known harmonic intervals.
///      This is aggressive but correct: a single piano note's harmonics in the mel
///      domain can be >50% of the fundamental, which is still a spurious detection.
fn suppress_harmonics_and_leakage(energy: &mut [f32; N_PITCHES], raw: &[f32; N_PITCHES]) {
    let min_floor: f32 = 0.5;

    // ---- Stage 1: Absolute floor ----
    for p in PIANO_LO as usize..=PIANO_HI as usize {
        if raw[p] < min_floor {
            energy[p] = 0.0;
        }
    }

    // ---- Stage 2: Local-maxima filter ----
    // A pitch is a local max if it has the highest raw energy within ±2 semitones.
    for p in PIANO_LO as usize..=PIANO_HI as usize {
        if energy[p] == 0.0 {
            continue;
        }
        let lo = if p >= PIANO_LO as usize + 2 { p - 2 } else { PIANO_LO as usize };
        let hi = if p + 2 <= PIANO_HI as usize { p + 2 } else { PIANO_HI as usize };
        let mut is_max = true;
        for q in lo..=hi {
            if q != p && raw[q] > raw[p] {
                is_max = false;
                break;
            }
        }
        if !is_max {
            energy[p] = 0.0;
        }
    }

    // ---- Stage 3: Harmonic masking (unconditional, bi-directional) ----
    // Harmonic intervals in semitones from fundamental:
    //   +12 (octave/2nd harmonic), +19 (octave+fifth/3rd),
    //   +24 (2 octaves/4th), +28 (2 oct+M3/5th), +31 (2 oct+P5/6th),
    //   +36 (3 octaves/8th).
    // Also suppress sub-harmonics (negative offsets) — a weaker pitch 12/19/24
    // semitones below a strong pitch is likely a phantom sub-harmonic from
    // interference patterns.
    const HARM_OFFSETS: &[i32] = &[
        12, 19, 24, 28, 31, 36,     // upward harmonics
        -12, -19, -24, -28, -31, -36, // sub-harmonics
    ];

    // Process from strongest to weakest
    let mut sorted: Vec<usize> = (PIANO_LO as usize..=PIANO_HI as usize)
        .filter(|&p| energy[p] > min_floor)
        .collect();
    sorted.sort_by(|&a, &b| energy[b].partial_cmp(&energy[a]).unwrap());

    let mut suppressed = [false; N_PITCHES];

    for &fund in &sorted {
        if suppressed[fund] {
            continue;
        }

        for &offset in HARM_OFFSETS {
            let h = fund as i32 + offset;
            if h < PIANO_LO as i32 || h > PIANO_HI as i32 {
                continue;
            }
            let hi = h as usize;
            if suppressed[hi] || hi == fund {
                continue;
            }
            // Unconditionally suppress: the harmonic of a stronger fundamental
            energy[hi] = 0.0;
            suppressed[hi] = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Mel-bin → MIDI pitch mapping
// ---------------------------------------------------------------------------

fn build_bin_to_pitch(n_mels: usize, mel_fmin: f32, mel_fmax: f32) -> Vec<Option<u8>> {
    let mel_lo = dsp::hz_to_mel(mel_fmin);
    let mel_hi = dsp::hz_to_mel(mel_fmax);

    (0..n_mels)
        .map(|bin| {
            let centre_mel =
                mel_lo + (mel_hi - mel_lo) * (bin + 1) as f32 / (n_mels + 1) as f32;
            let hz = dsp::mel_to_hz(centre_mel);
            if hz < 27.0 {
                return None;
            }
            let midi_f = 69.0 + 12.0 * (hz / 440.0).log2();
            let midi = midi_f.round() as i32;
            if midi >= PIANO_LO as i32 && midi <= PIANO_HI as i32 {
                Some(midi as u8)
            } else {
                None
            }
        })
        .collect()
}
