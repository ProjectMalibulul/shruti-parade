use crossbeam_channel::{Receiver, Sender};
use std::collections::HashSet;
use tracing::{debug, info};

use crate::config::InferenceConfig;
use crate::dsp::{self, PitchFrame};
use crate::events::{NoteEvent, NoteEventKind};

const PIANO_LO: u8 = 21; // A0
const PIANO_HI: u8 = 108; // C8
const N_PITCHES: usize = 128;

/// Maximum simultaneous notes — prevents noise from triggering the full keyboard.
const MAX_POLYPHONY: usize = 10;

/// Energy + flux onset detector operating on harmonic-sieve pitch energies.
///
/// The heavy lifting of pitch detection (harmonic partial weighting,
/// spectral-leakage handling) is done upstream in the DSP thread via
/// `compute_pitch_energies`.  This engine just does onset/offset tracking.
pub struct InferenceEngine {
    config: InferenceConfig,
    pitch_rx: Receiver<PitchFrame>,
    event_tx: Sender<NoteEvent>,
}

impl InferenceEngine {
    pub fn new(
        config: InferenceConfig,
        pitch_rx: Receiver<PitchFrame>,
        event_tx: Sender<NoteEvent>,
    ) -> Self {
        Self {
            config,
            pitch_rx,
            event_tx,
        }
    }

    /// Blocking onset-detection loop — call from a dedicated thread.
    pub fn run(&mut self) {
        info!("Inference engine started (harmonic-sieve onset detector)");

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
        let mut events_sent: u64 = 0;

        // Sustain pedal simulation: detect when multiple notes overlap
        // (typical of pedaled passages) and extend their release window.
        let mut sustained_notes: HashSet<u8> = HashSet::new();
        let mut active_frames = [0u32; N_PITCHES];
        // When ≥3 notes overlap for ≥4 frames, engage simulated pedal
        const PEDAL_OVERLAP_THRESHOLD: usize = 3;
        const PEDAL_ENGAGE_FRAMES: u32 = 4;
        // Extended release when pedal is engaged
        const SUSTAINED_RELEASE_FRAMES: u32 = 24;

        // Noise floor starts very low — warmup will calibrate it
        for nf in noise_floor.iter_mut() {
            *nf = 0.001;
        }

        // ---- Tuning constants ----
        // The harmonic sieve produces weighted arithmetic-mean magnitudes.
        // For a 4096-point FFT of a moderate piano note (amplitude ~0.3),
        // typical pitch energy ≈ 100–600. Silence ≈ 0–5.
        let abs_min_energy: f32 = 4.0;
        // SNR: energy must exceed noise floor by this factor for onset
        let snr_onset: f32 = 4.0 + self.config.onset_threshold * 6.0; // default: 7.0
                                                                      // Minimum frame-over-frame flux ratio on first detection frame
        let flux_ratio: f32 = 1.5;
        // Noise floor adaptation
        let nf_rise: f32 = 0.003;
        let nf_fall: f32 = 0.06;
        // Onset confirmation: require N consecutive frames above threshold
        let onset_confirm: u8 = 2;
        // Offset parameters
        let release_ratio: f32 = 2.0 + self.config.frame_threshold * 2.0;
        let release_frames: u32 = 4;
        let peak_decay_ratio: f32 = 5.0;
        // Cooldown after NoteOff before the same pitch can retrigger
        let retrigger_cooldown: u32 = 3;

        let mut frame_num: u64 = 0;
        let mut warmup_done = false;

        for frame in self.pitch_rx.iter() {
            let sample_time = frame.sample_offset;
            let pitch_energy = frame.pitch_energy;
            frame_num += 1;

            // ---- 1. Harmonic aliasing suppression ----
            // Suppress pitches that are harmonics of stronger pitches.
            // Ratio threshold 0.35: keep only if energy ≥ 35% of the fundamental.
            let mut filtered = pitch_energy;
            dsp::suppress_harmonic_aliasing(&mut filtered, 0.35);

            // ---- 2. Local-max filter ±2 semitones ----
            // After aliasing suppression, remove pitches that aren't local peaks.
            for p in PIANO_LO as usize..=PIANO_HI as usize {
                if filtered[p] < abs_min_energy {
                    filtered[p] = 0.0;
                    continue;
                }
                let lo = if p >= PIANO_LO as usize + 2 {
                    p - 2
                } else {
                    PIANO_LO as usize
                };
                let hi = if p + 2 <= PIANO_HI as usize {
                    p + 2
                } else {
                    PIANO_HI as usize
                };
                let mut is_max = true;
                for q in lo..=hi {
                    if q != p && filtered[q] > filtered[p] {
                        is_max = false;
                        break;
                    }
                }
                if !is_max {
                    filtered[p] = 0.0;
                }
            }

            // ---- 2. Warmup: calibrate noise floor from first 30 frames ----
            if frame_num <= 30 {
                for i in PIANO_LO as usize..=PIANO_HI as usize {
                    noise_floor[i] += 0.2 * (filtered[i] - noise_floor[i]);
                }
                prev_energy[..N_PITCHES].copy_from_slice(&filtered[..N_PITCHES]);
                continue;
            }

            if !warmup_done {
                warmup_done = true;
                for nf in &mut noise_floor[PIANO_LO as usize..=PIANO_HI as usize] {
                    *nf *= 1.5;
                    if *nf < 1.0 {
                        *nf = 1.0;
                    }
                }
                info!("Noise floor calibrated (30 frames)");
            }

            // ---- 3. Per-pitch onset / offset ----
            // Detect pedal engagement: many overlapping long-held notes
            let long_active = (PIANO_LO as usize..=PIANO_HI as usize)
                .filter(|&i| active[i] && active_frames[i] >= PEDAL_ENGAGE_FRAMES)
                .count();
            let pedal_held = long_active >= PEDAL_OVERLAP_THRESHOLD;

            for pitch in PIANO_LO..=PIANO_HI {
                let i = pitch as usize;
                let e = filtered[i];
                let prev = prev_energy[i];

                // Tick cooldown
                if cooldown[i] > 0 {
                    cooldown[i] -= 1;
                }

                // Update noise floor (only from inactive, non-pending pitches)
                if !active[i] && onset_pending[i] == 0 {
                    let alpha = if e > noise_floor[i] { nf_rise } else { nf_fall };
                    noise_floor[i] += alpha * (e - noise_floor[i]);
                    if noise_floor[i] < 1.0 {
                        noise_floor[i] = 1.0;
                    }
                }

                let onset_thresh = (noise_floor[i] * snr_onset).max(abs_min_energy);

                // Flux: current / previous ratio.
                // When prev is near-zero and energy is above minimum → large onset.
                let flux = if prev < 1.0 {
                    if e > abs_min_energy {
                        f32::INFINITY
                    } else {
                        0.0
                    }
                } else {
                    e / prev
                };

                if !active[i] {
                    if onset_pending[i] > 0 {
                        // Already pending: only need sustained energy
                        if e > onset_thresh && cooldown[i] == 0 {
                            onset_pending[i] += 1;
                            if onset_pending[i] >= onset_confirm {
                                if active_count >= MAX_POLYPHONY {
                                    onset_pending[i] = 0;
                                } else {
                                    active[i] = true;
                                    onset_pending[i] = 0;
                                    peak_energy[i] = e;
                                    below_count[i] = 0;
                                    active_frames[i] = 0;
                                    active_count += 1;

                                    let _ = self.event_tx.send(NoteEvent {
                                        kind: NoteEventKind::NoteOn,
                                        pitch,
                                        velocity: pending_vel[i],
                                        sample_time,
                                    });
                                    events_sent += 1;
                                    if events_sent == 1 {
                                        info!(
                                            "Inference: first NoteOn pitch={} vel={} at sample {}",
                                            pitch, pending_vel[i], sample_time
                                        );
                                    }
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
                    active_frames[i] += 1;

                    // ---- Retrigger detection: sharp flux on active note ----
                    // If energy suddenly spikes well above the current peak,
                    // the same key was re-struck. End current note, start new.
                    let retrigger_flux: f32 = 2.5;
                    if active_frames[i] > 4 && flux > retrigger_flux && e > onset_thresh {
                        // End current note
                        let _ = self.event_tx.send(NoteEvent {
                            kind: NoteEventKind::NoteOff,
                            pitch,
                            velocity: 0,
                            sample_time,
                        });
                        // Immediately start new note
                        peak_energy[i] = e;
                        below_count[i] = 0;
                        active_frames[i] = 0;
                        let ratio = e / onset_thresh;
                        let vel = ((ratio.sqrt() * 20.0 + 40.0).clamp(40.0, 127.0)) as u8;
                        let _ = self.event_tx.send(NoteEvent {
                            kind: NoteEventKind::NoteOn,
                            pitch,
                            velocity: vel,
                            sample_time,
                        });
                        events_sent += 1;
                        prev_energy[i] = e;
                        continue;
                    }

                    if e > peak_energy[i] {
                        peak_energy[i] = e;
                    }

                    let off_thresh = (noise_floor[i] * release_ratio).max(abs_min_energy * 0.5);
                    let peak_drop = peak_energy[i] / peak_decay_ratio;

                    // Use extended release window when sustain pedal is engaged
                    let effective_release = if pedal_held {
                        SUSTAINED_RELEASE_FRAMES
                    } else {
                        release_frames
                    };

                    if e < off_thresh || e < peak_drop {
                        below_count[i] += 1;
                        if below_count[i] >= effective_release {
                            active[i] = false;
                            active_frames[i] = 0;
                            active_count = active_count.saturating_sub(1);
                            cooldown[i] = retrigger_cooldown;
                            sustained_notes.remove(&pitch);
                            let _ = self.event_tx.send(NoteEvent {
                                kind: NoteEventKind::NoteOff,
                                pitch,
                                velocity: 0,
                                sample_time,
                            });
                        }
                    } else {
                        below_count[i] = 0;
                        if pedal_held {
                            sustained_notes.insert(pitch);
                        }
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

#[cfg(test)]
mod inference_tests {
    use super::*;
    use crate::config::InferenceConfig;
    use crate::dsp::PitchFrame;

    fn default_inference_config() -> InferenceConfig {
        InferenceConfig {
            model_path: String::new(),
            context_frames: 32,
            overlap_frames: 8,
            onset_threshold: 0.5,
            frame_threshold: 0.3,
        }
    }

    #[test]
    fn offset_after_silence() {
        let (pitch_tx, pitch_rx) = crossbeam_channel::unbounded();
        let (event_tx, event_rx) = crossbeam_channel::unbounded();
        let config = default_inference_config();

        let handle = std::thread::spawn(move || {
            let mut engine = InferenceEngine::new(config, pitch_rx, event_tx);
            engine.run();
        });

        let target_pitch: u8 = 60; // C4

        // Send 30 warmup frames (silent) to calibrate noise floor
        for i in 0..30 {
            let frame = PitchFrame {
                pitch_energy: [0.0; 128],
                sample_offset: i * 512,
            };
            pitch_tx.send(frame).unwrap();
        }

        // Send frames with high energy for note 60 (onset)
        for i in 30..50 {
            let mut energy = [0.0f32; 128];
            energy[target_pitch as usize] = 500.0;
            let frame = PitchFrame {
                pitch_energy: energy,
                sample_offset: i * 512,
            };
            pitch_tx.send(frame).unwrap();
        }

        // Send zero-energy frames (should trigger offset)
        for i in 50..80 {
            let frame = PitchFrame {
                pitch_energy: [0.0; 128],
                sample_offset: i * 512,
            };
            pitch_tx.send(frame).unwrap();
        }

        // Close channel to stop engine
        drop(pitch_tx);
        handle.join().unwrap();

        // Collect all events
        let events: Vec<NoteEvent> = event_rx.try_iter().collect();

        let has_note_off = events
            .iter()
            .any(|e| e.kind == NoteEventKind::NoteOff && e.pitch == target_pitch);
        assert!(
            has_note_off,
            "Expected NoteOff for pitch {target_pitch}, got events: {events:?}"
        );
    }

    #[test]
    fn test_sustain_pedal_holds_note() {
        // When ≥3 notes overlap, the simulated sustain pedal should engage
        // and extend the release window, delaying NoteOff.
        let (pitch_tx, pitch_rx) = crossbeam_channel::unbounded();
        let (event_tx, event_rx) = crossbeam_channel::unbounded();
        let config = default_inference_config();

        let handle = std::thread::spawn(move || {
            let mut engine = InferenceEngine::new(config, pitch_rx, event_tx);
            engine.run();
        });

        // 30 warmup frames
        for i in 0..30 {
            pitch_tx
                .send(PitchFrame {
                    pitch_energy: [0.0; 128],
                    sample_offset: i * 512,
                })
                .unwrap();
        }

        // Onset 3 notes simultaneously (C4=60, E4=64, G4=67) for 20 frames
        for i in 30..50 {
            let mut energy = [0.0f32; 128];
            energy[60] = 500.0;
            energy[64] = 400.0;
            energy[67] = 350.0;
            pitch_tx
                .send(PitchFrame {
                    pitch_energy: energy,
                    sample_offset: i * 512,
                })
                .unwrap();
        }

        // Drop energy for C4 only, but keep E4+G4 active
        // With only standard release_frames=8, C4 would turn off quickly.
        // With pedal engaged (3 overlapping notes), release extends to 24.
        for i in 50..60 {
            let mut energy = [0.0f32; 128];
            energy[60] = 0.0; // C4 dropped
            energy[64] = 400.0;
            energy[67] = 350.0;
            pitch_tx
                .send(PitchFrame {
                    pitch_energy: energy,
                    sample_offset: i * 512,
                })
                .unwrap();
        }

        // At frame 60 (10 frames after drop), C4 should still be held by pedal
        // because sustained release is 24 frames
        // Send enough frames for C4 to eventually release
        for i in 60..90 {
            let mut energy = [0.0f32; 128];
            energy[64] = 400.0;
            energy[67] = 350.0;
            pitch_tx
                .send(PitchFrame {
                    pitch_energy: energy,
                    sample_offset: i * 512,
                })
                .unwrap();
        }

        drop(pitch_tx);
        handle.join().unwrap();

        let events: Vec<NoteEvent> = event_rx.try_iter().collect();

        // All three notes should have been onset
        let onsets: Vec<u8> = events
            .iter()
            .filter(|e| e.kind == NoteEventKind::NoteOn)
            .map(|e| e.pitch)
            .collect();
        assert!(onsets.contains(&60), "C4 should have onset");
        assert!(onsets.contains(&64), "E4 should have onset");
        assert!(onsets.contains(&67), "G4 should have onset");

        // C4 should eventually get NoteOff (after extended sustain)
        let c4_off = events
            .iter()
            .find(|e| e.kind == NoteEventKind::NoteOff && e.pitch == 60);
        assert!(
            c4_off.is_some(),
            "C4 should eventually get NoteOff after sustain release"
        );

        // Verify that C4's NoteOff came after frame 58 (i.e., was held
        // beyond the standard 8-frame release window)
        if let Some(off_event) = c4_off {
            let off_frame = off_event.sample_time / 512;
            assert!(
                off_frame > 58,
                "C4 NoteOff at frame {off_frame} — expected >58 (sustain should delay release)"
            );
        }
    }
}
