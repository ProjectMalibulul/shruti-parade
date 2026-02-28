/// Diagnostic test: trace a single-note signal through DSP → mel → pitch mapping
/// to understand energy distribution and harmonic bleed.
///
/// This test does NOT need GPU or audio hardware.

use realfft::RealFftPlanner;
use shruti_parade::dsp;

const SR: f32 = 48000.0;
const FFT_SIZE: usize = 2048;
const N_MELS: usize = 229;
const MEL_FMIN: f32 = 30.0;
const MEL_FMAX: f32 = 8000.0;

const PIANO_LO: u8 = 21;
const PIANO_HI: u8 = 108;

fn midi_to_hz(midi: u8) -> f32 {
    440.0 * 2f32.powf((midi as f32 - 69.0) / 12.0)
}

/// Mirror of the suppression function in inference.rs for testing
fn suppress_harmonics_and_leakage(energy: &mut [f32; 128], raw: &[f32; 128]) {
    let min_floor: f32 = 0.5;

    // Stage 1: Absolute floor
    for p in PIANO_LO as usize..=PIANO_HI as usize {
        if raw[p] < min_floor {
            energy[p] = 0.0;
        }
    }

    // Stage 2: Local-maxima filter
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

    // Stage 3: Unconditional harmonic masking (bi-directional)
    const HARM_OFFSETS: &[i32] = &[
        12, 19, 24, 28, 31, 36,
        -12, -19, -24, -28, -31, -36,
    ];

    let mut sorted: Vec<usize> = (PIANO_LO as usize..=PIANO_HI as usize)
        .filter(|&p| energy[p] > min_floor)
        .collect();
    sorted.sort_by(|&a, &b| energy[b].partial_cmp(&energy[a]).unwrap());

    let mut suppressed = [false; 128];

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
            energy[hi] = 0.0;
            suppressed[hi] = true;
        }
    }
}

/// Build the same mel→pitch map as inference.rs
fn build_bin_to_pitch() -> Vec<Option<u8>> {
    let mel_lo = dsp::hz_to_mel(MEL_FMIN);
    let mel_hi = dsp::hz_to_mel(MEL_FMAX);

    (0..N_MELS)
        .map(|bin| {
            let centre_mel =
                mel_lo + (mel_hi - mel_lo) * (bin + 1) as f32 / (N_MELS + 1) as f32;
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

/// Generate a piano-like signal: fundamental + harmonics with natural decay
fn generate_piano_signal(midi: u8, n_samples: usize) -> Vec<f32> {
    let f0 = midi_to_hz(midi);
    let mut out = vec![0.0f32; n_samples];
    // Piano harmonics with natural amplitude decay
    let harmonics: &[(f32, f32)] = &[
        (1.0, 1.0),   // fundamental
        (2.0, 0.5),   // 2nd harmonic
        (3.0, 0.25),  // 3rd
        (4.0, 0.12),  // 4th
        (5.0, 0.06),  // 5th
        (6.0, 0.03),  // 6th
    ];
    for (i, s) in out.iter_mut().enumerate() {
        let t = i as f32 / SR;
        for &(h, amp) in harmonics {
            let freq = f0 * h;
            if freq < SR / 2.0 {
                *s += amp * (2.0 * std::f32::consts::PI * freq * t).sin();
            }
        }
    }
    out
}

/// Run FFT + mel filterbank on a frame, return log-domain mel energies
fn compute_mel_frame(signal: &[f32], mel_fb: &[Vec<f32>]) -> Vec<f32> {
    assert_eq!(signal.len(), FFT_SIZE);

    // Hann window
    let mut windowed = vec![0.0f32; FFT_SIZE];
    for (i, s) in signal.iter().enumerate() {
        let w = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / FFT_SIZE as f32).cos());
        windowed[i] = s * w;
    }

    // FFT
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);
    let mut scratch = fft.make_scratch_vec();
    let n_bins = FFT_SIZE / 2 + 1;
    let mut fft_out = vec![rustfft::num_complex::Complex::default(); n_bins];
    fft.process_with_scratch(&mut windowed, &mut fft_out, &mut scratch)
        .unwrap();

    // Mel filterbank → log
    mel_fb
        .iter()
        .map(|filt| {
            let energy: f32 = filt
                .iter()
                .zip(fft_out.iter())
                .map(|(&w, c)| w * (c.re * c.re + c.im * c.im).sqrt())
                .sum();
            (energy.max(1e-10)).ln()
        })
        .collect()
}

/// Map mel energies → per-pitch energies (same as inference.rs)
fn mel_to_pitch_energy(mel: &[f32], btp: &[Option<u8>]) -> [f32; 128] {
    let mut pitch_energy = [0.0f32; 128];
    for (bin, &log_e) in mel.iter().enumerate() {
        if let Some(&Some(midi)) = btp.get(bin) {
            let linear = log_e.exp();
            if linear > pitch_energy[midi as usize] {
                pitch_energy[midi as usize] = linear;
            }
        }
    }
    pitch_energy
}

#[test]
fn diagnose_single_note_energy_distribution() {
    let mel_fb = dsp::build_mel_filterbank(N_MELS, FFT_SIZE, SR, MEL_FMIN, MEL_FMAX);
    let btp = build_bin_to_pitch();

    // ---- Test with A4 (MIDI 69, 440 Hz) ----
    let target_midi: u8 = 69;
    let signal = generate_piano_signal(target_midi, FFT_SIZE);
    let mel = compute_mel_frame(&signal, &mel_fb);
    let pe = mel_to_pitch_energy(&mel, &btp);

    println!("\n=== SINGLE NOTE: A4 (MIDI 69, 440 Hz) ===");
    println!("Target pitch {target_midi} energy: {:.4}", pe[target_midi as usize]);

    // Show ALL pitches with non-trivial energy
    let mut energies: Vec<(u8, f32)> = (PIANO_LO..=PIANO_HI)
        .map(|p| (p, pe[p as usize]))
        .filter(|(_, e)| *e > 0.001)
        .collect();
    energies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nAll pitches with energy > 0.001 (sorted by energy):");
    for (pitch, e) in &energies {
        let hz = midi_to_hz(*pitch);
        let ratio = e / pe[target_midi as usize];
        println!(
            "  MIDI {:3} ({:7.1} Hz): energy = {:8.4}, ratio to target = {:.4}",
            pitch, hz, e, ratio
        );
    }

    let target_e = pe[target_midi as usize];
    let significant = energies.iter().filter(|(_, e)| *e > target_e * 0.1).count();
    println!("\nPitches with >10% of target energy: {significant}");
    let spurious = energies
        .iter()
        .filter(|(p, e)| *p != target_midi && *e > target_e * 0.1)
        .count();
    println!("Spurious pitches (>10% of target, wrong pitch): {spurious}");

    // ---- Test with silence ----
    let silence = vec![0.0f32; FFT_SIZE];
    let mel_silence = compute_mel_frame(&silence, &mel_fb);
    let pe_silence = mel_to_pitch_energy(&mel_silence, &btp);
    let max_silence_e = (PIANO_LO..=PIANO_HI)
        .map(|p| pe_silence[p as usize])
        .fold(0.0f32, f32::max);
    println!("\n=== SILENCE ===");
    println!("Max pitch energy in silence: {:.10}", max_silence_e);

    // ---- Test with noise ----
    let noise: Vec<f32> = (0..FFT_SIZE)
        .map(|i| {
            // Deterministic pseudo-noise
            let x = (i as f32 * 0.7534 + 0.123).sin() * 0.01;
            x
        })
        .collect();
    let mel_noise = compute_mel_frame(&noise, &mel_fb);
    let pe_noise = mel_to_pitch_energy(&mel_noise, &btp);
    let max_noise_e = (PIANO_LO..=PIANO_HI)
        .map(|p| pe_noise[p as usize])
        .fold(0.0f32, f32::max);
    println!("\n=== LOW-LEVEL NOISE (amplitude 0.01) ===");
    println!("Max pitch energy: {:.6}", max_noise_e);
    let noisy_count = (PIANO_LO..=PIANO_HI)
        .filter(|&p| pe_noise[p as usize] > 0.05)
        .count();
    println!("Pitches with energy > 0.05 (abs_min): {noisy_count}");

    // ---- Simulate onset detection for the single note ----
    // Frame 1: silence, Frame 2: note onset
    println!("\n=== ONSET DETECTION SIMULATION (v4 logic) ===");
    let abs_min_energy: f32 = 0.5;
    let snr_onset: f32 = 10.0;
    let flux_ratio: f32 = 1.8;

    // Apply harmonic suppression to the note energy
    let mut suppressed_energy = pe;
    suppress_harmonics_and_leakage(&mut suppressed_energy, &pe);

    println!("\nAfter harmonic/leakage suppression:");
    let mut supe: Vec<(u8, f32)> = (PIANO_LO..=PIANO_HI)
        .map(|p| (p, suppressed_energy[p as usize]))
        .filter(|(_, e)| *e > 0.001)
        .collect();
    supe.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (pitch, e) in &supe {
        let hz = midi_to_hz(*pitch);
        println!(
            "  MIDI {:3} ({:7.1} Hz): energy = {:8.4}",
            pitch, hz, e
        );
    }
    println!("Pitches with energy >0.001 after suppression: {}", supe.len());

    // Calibrated noise floor from silence
    let mut noise_floor = [0.005f32; 128];
    for p in PIANO_LO..=PIANO_HI {
        noise_floor[p as usize] = pe_silence[p as usize].max(0.005);
    }

    // Now check which pitches would trigger onset with FIXED flux logic
    let mut triggered = Vec::new();
    for p in PIANO_LO..=PIANO_HI {
        let i = p as usize;
        let e = suppressed_energy[i];
        let prev = pe_silence[i]; // previous frame was silence
        let thresh = (noise_floor[i] * snr_onset).max(abs_min_energy);
        // FIXED: when prev is near-zero and energy is significant, 
        // that's a huge onset — treat as infinite flux
        let flux = if prev < 1e-4 {
            if e > abs_min_energy { f32::INFINITY } else { 0.0 }
        } else {
            e / prev
        };

        if e > thresh && flux > flux_ratio {
            triggered.push((p, e, thresh, flux));
        }
    }

    println!("\nPitches that would trigger onset (v4 logic, with suppression):");
    for (p, e, thresh, flux) in &triggered {
        let hz = midi_to_hz(*p);
        let flux_str = if flux.is_infinite() { "INF".to_string() } else { format!("{:.1}", flux) };
        println!(
            "  MIDI {:3} ({:7.1} Hz): energy={:.4}, thresh={:.4}, flux={}",
            p, hz, e, thresh, flux_str
        );
    }
    println!("Total triggered: {}", triggered.len());

    // The assertion: a single note should trigger at most a few pitches
    println!("\n=== SUMMARY ===");
    println!("For a threshold system to work, we need triggered <= ~3-4 for a single note.");
    println!("Actual triggered: {}", triggered.len());

    assert_eq!(triggered.len(), 1, "Single note should trigger exactly 1 pitch, got {}", triggered.len());
    assert_eq!(triggered[0].0, target_midi, "Triggered pitch should be target {}", target_midi);
    
    println!("PASS: exactly 1 pitch triggered (MIDI {})", triggered[0].0);
}

/// Test with two simultaneous notes (C4 + E4 = major third).
/// Both should survive suppression since they're not at harmonic intervals.
#[test]
fn diagnose_two_simultaneous_notes() {
    let mel_fb = dsp::build_mel_filterbank(N_MELS, FFT_SIZE, SR, MEL_FMIN, MEL_FMAX);
    let btp = build_bin_to_pitch();

    let note_a: u8 = 60; // C4
    let note_b: u8 = 64; // E4

    // Generate combined signal
    let sig_a = generate_piano_signal(note_a, FFT_SIZE);
    let sig_b = generate_piano_signal(note_b, FFT_SIZE);
    let combined: Vec<f32> = sig_a.iter().zip(sig_b.iter()).map(|(a, b)| a + b).collect();

    let mel = compute_mel_frame(&combined, &mel_fb);
    let pe = mel_to_pitch_energy(&mel, &btp);
    let mut suppressed = pe;
    suppress_harmonics_and_leakage(&mut suppressed, &pe);

    let abs_min_energy: f32 = 0.5;
    let snr_onset: f32 = 10.0;
    let flux_ratio: f32 = 1.8;
    let silence = vec![0.0f32; FFT_SIZE];
    let mel_silence = compute_mel_frame(&silence, &mel_fb);
    let pe_silence = mel_to_pitch_energy(&mel_silence, &btp);

    let mut noise_floor = [0.005f32; 128];
    for p in PIANO_LO..=PIANO_HI {
        noise_floor[p as usize] = pe_silence[p as usize].max(0.005);
    }

    let mut triggered = Vec::new();
    for p in PIANO_LO..=PIANO_HI {
        let i = p as usize;
        let e = suppressed[i];
        let prev = pe_silence[i];
        let thresh = (noise_floor[i] * snr_onset).max(abs_min_energy);
        let flux = if prev < 1e-4 {
            if e > abs_min_energy { f32::INFINITY } else { 0.0 }
        } else {
            e / prev
        };
        if e > thresh && flux > flux_ratio {
            triggered.push(p);
        }
    }

    println!("\n=== TWO SIMULTANEOUS NOTES: C4 ({}) + E4 ({}) ===", note_a, note_b);
    println!("Triggered pitches: {:?}", triggered);
    println!("Count: {}", triggered.len());

    // Both target notes should be present
    assert!(triggered.contains(&note_a), "C4 (MIDI {}) should be triggered", note_a);
    assert!(triggered.contains(&note_b), "E4 (MIDI {}) should be triggered", note_b);
    // Should be at most 3 (2 targets + maybe 1 stray)
    assert!(triggered.len() <= 3, "Too many pitches: {:?}", triggered);

    println!("PASS: {} pitches triggered, both targets present", triggered.len());
}

/// Test with three simultaneous notes (C major chord: C4 + E4 + G4).
#[test]
fn diagnose_three_simultaneous_notes() {
    let mel_fb = dsp::build_mel_filterbank(N_MELS, FFT_SIZE, SR, MEL_FMIN, MEL_FMAX);
    let btp = build_bin_to_pitch();

    let notes: &[u8] = &[60, 64, 67]; // C4, E4, G4

    let mut combined = vec![0.0f32; FFT_SIZE];
    for &midi in notes {
        let sig = generate_piano_signal(midi, FFT_SIZE);
        for (i, s) in sig.iter().enumerate() {
            combined[i] += s;
        }
    }

    let mel = compute_mel_frame(&combined, &mel_fb);
    let pe = mel_to_pitch_energy(&mel, &btp);
    let mut suppressed = pe;
    suppress_harmonics_and_leakage(&mut suppressed, &pe);

    let abs_min_energy: f32 = 0.5;
    let snr_onset: f32 = 10.0;
    let flux_ratio: f32 = 1.8;
    let silence = vec![0.0f32; FFT_SIZE];
    let mel_silence = compute_mel_frame(&silence, &mel_fb);
    let pe_silence = mel_to_pitch_energy(&mel_silence, &btp);

    let mut noise_floor = [0.005f32; 128];
    for p in PIANO_LO..=PIANO_HI {
        noise_floor[p as usize] = pe_silence[p as usize].max(0.005);
    }

    let mut triggered = Vec::new();
    for p in PIANO_LO..=PIANO_HI {
        let i = p as usize;
        let e = suppressed[i];
        let prev = pe_silence[i];
        let thresh = (noise_floor[i] * snr_onset).max(abs_min_energy);
        let flux = if prev < 1e-4 {
            if e > abs_min_energy { f32::INFINITY } else { 0.0 }
        } else {
            e / prev
        };
        if e > thresh && flux > flux_ratio {
            triggered.push(p);
        }
    }

    println!("\n=== C MAJOR CHORD: C4 (60) + E4 (64) + G4 (67) ===");
    println!("Triggered pitches: {:?}", triggered);
    println!("Count: {}", triggered.len());

    for &n in notes {
        assert!(triggered.contains(&n), "MIDI {} should be triggered", n);
    }
    assert!(triggered.len() <= 5, "Too many pitches: {:?}", triggered);

    println!("PASS: {} pitches triggered, all 3 targets present", triggered.len());
}

#[test]
fn diagnose_bin_to_pitch_mapping() {
    let btp = build_bin_to_pitch();

    println!("\n=== MEL BIN → MIDI PITCH MAPPING ===");

    // Count how many bins map to each pitch
    let mut bins_per_pitch = [0u32; 128];
    for opt in &btp {
        if let Some(midi) = opt {
            bins_per_pitch[*midi as usize] += 1;
        }
    }

    println!("Bins per pitch (MIDI 21-108):");
    for p in PIANO_LO..=PIANO_HI {
        if bins_per_pitch[p as usize] > 0 {
            let hz = midi_to_hz(p);
            println!(
                "  MIDI {:3} ({:7.1} Hz): {} bins",
                p, hz, bins_per_pitch[p as usize]
            );
        }
    }

    let covered = (PIANO_LO..=PIANO_HI)
        .filter(|&p| bins_per_pitch[p as usize] > 0)
        .count();
    let total = (PIANO_HI - PIANO_LO + 1) as usize;
    println!(
        "\nCovered: {covered}/{total} pitches ({:.0}%)",
        covered as f32 / total as f32 * 100.0
    );

    let unmapped: Vec<u8> = (PIANO_LO..=PIANO_HI)
        .filter(|&p| bins_per_pitch[p as usize] == 0)
        .collect();
    if !unmapped.is_empty() {
        println!("Unmapped pitches: {:?}", unmapped);
    }
}
