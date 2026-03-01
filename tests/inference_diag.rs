/// Diagnostic tests for the harmonic-sieve DSP → pitch energy pipeline.
///
/// These tests generate synthetic piano-like signals, run FFT + harmonic sieve,
/// and verify correct pitch detection.
///
/// No GPU or audio hardware required.
use realfft::RealFftPlanner;
use shruti_parade::dsp;

const SR: f32 = 48000.0;
const FFT_SIZE: usize = 4096;

const PIANO_LO: u8 = 21;
const PIANO_HI: u8 = 108;

fn midi_to_hz(midi: u8) -> f32 {
    440.0 * 2f32.powf((midi as f32 - 69.0) / 12.0)
}

/// Generate a piano-like signal: fundamental + harmonics with natural decay.
fn generate_piano_signal(midi: u8, n_samples: usize) -> Vec<f32> {
    let f0 = midi_to_hz(midi);
    let mut out = vec![0.0f32; n_samples];
    let harmonics: &[(f32, f32)] = &[
        (1.0, 1.0),  // fundamental
        (2.0, 0.5),  // 2nd harmonic
        (3.0, 0.25), // 3rd
        (4.0, 0.12), // 4th
        (5.0, 0.06), // 5th
        (6.0, 0.03), // 6th
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

/// Compute FFT magnitudes from a windowed signal frame.
fn compute_fft_magnitudes(signal: &[f32]) -> Vec<f32> {
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

    // Magnitudes
    fft_out
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .collect()
}

#[test]
fn diagnose_single_note_energy_distribution() {
    let templates = dsp::build_harmonic_templates(SR, FFT_SIZE);

    // ---- Test with A4 (MIDI 69, 440 Hz) ----
    let target_midi: u8 = 69;
    let signal = generate_piano_signal(target_midi, FFT_SIZE);
    let magnitudes = compute_fft_magnitudes(&signal);
    let pitch_energy = dsp::compute_pitch_energies(&magnitudes, &templates);

    println!("\n=== SINGLE NOTE: A4 (MIDI 69, 440 Hz) ===");
    println!(
        "Target pitch {target_midi} energy: {:.4}",
        pitch_energy[target_midi as usize]
    );

    // Show ALL pitches with non-trivial energy
    let mut energies: Vec<(u8, f32)> = (PIANO_LO..=PIANO_HI)
        .map(|p| (p, pitch_energy[p as usize]))
        .filter(|(_, e)| *e > 1.0)
        .collect();
    energies.sort_by(|a, b| b.1.total_cmp(&a.1));

    println!("\nAll pitches with energy > 1.0 (sorted):");
    for (pitch, e) in &energies {
        let hz = midi_to_hz(*pitch);
        let ratio = e / pitch_energy[target_midi as usize];
        println!(
            "  MIDI {:3} ({:7.1} Hz): energy = {:8.2}, ratio = {:.4}",
            pitch, hz, e, ratio
        );
    }

    // Apply harmonic aliasing suppression + local-max filter (same as inference engine)
    let abs_min_energy: f32 = 4.0;
    let mut filtered = pitch_energy;
    dsp::suppress_harmonic_aliasing(&mut filtered, 0.50);

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

    let surviving: Vec<(u8, f32)> = (PIANO_LO..=PIANO_HI)
        .map(|p| (p, filtered[p as usize]))
        .filter(|(_, e)| *e > abs_min_energy)
        .collect();

    println!("\nAfter local-max filter (abs_min = {abs_min_energy}):");
    for (pitch, e) in &surviving {
        let hz = midi_to_hz(*pitch);
        println!("  MIDI {:3} ({:7.1} Hz): energy = {:8.2}", pitch, hz, e);
    }
    println!("Surviving pitches: {}", surviving.len());

    // The target should be present and be the strongest
    assert!(
        surviving.iter().any(|(p, _)| *p == target_midi),
        "Target MIDI {} must survive filtering",
        target_midi
    );

    // For a single note, expect very few surviving pitches (ideally 1)
    assert!(
        surviving.len() <= 3,
        "Too many surviving pitches for single note: {} (expected <= 3)",
        surviving.len()
    );

    println!(
        "PASS: {} surviving pitch(es), target MIDI {} present",
        surviving.len(),
        target_midi
    );
}

/// Test with two simultaneous notes (C4 + E4 = major third).
#[test]
fn diagnose_two_simultaneous_notes() {
    let templates = dsp::build_harmonic_templates(SR, FFT_SIZE);

    let note_a: u8 = 60; // C4
    let note_b: u8 = 64; // E4

    let sig_a = generate_piano_signal(note_a, FFT_SIZE);
    let sig_b = generate_piano_signal(note_b, FFT_SIZE);
    let combined: Vec<f32> = sig_a.iter().zip(sig_b.iter()).map(|(a, b)| a + b).collect();

    let magnitudes = compute_fft_magnitudes(&combined);
    let pitch_energy = dsp::compute_pitch_energies(&magnitudes, &templates);

    // Harmonic aliasing suppression + local-max ±2 semitones (same as inference)
    let abs_min_energy: f32 = 4.0;
    let mut filtered = pitch_energy;
    dsp::suppress_harmonic_aliasing(&mut filtered, 0.50);

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

    let surviving: Vec<u8> = (PIANO_LO..=PIANO_HI)
        .filter(|&p| filtered[p as usize] > abs_min_energy)
        .collect();

    println!("\n=== TWO NOTES: C4 ({}) + E4 ({}) ===", note_a, note_b);
    println!("Surviving pitches: {:?}", surviving);

    assert!(
        surviving.contains(&note_a),
        "C4 (MIDI {}) should survive",
        note_a
    );
    assert!(
        surviving.contains(&note_b),
        "E4 (MIDI {}) should survive",
        note_b
    );
    assert!(surviving.len() <= 4, "Too many pitches: {:?}", surviving);

    println!("PASS: {} pitches, both targets present", surviving.len());
}

/// Test with three simultaneous notes (C major chord: C4 + E4 + G4).
#[test]
fn diagnose_three_simultaneous_notes() {
    let templates = dsp::build_harmonic_templates(SR, FFT_SIZE);

    let notes: &[u8] = &[60, 64, 67]; // C4, E4, G4

    let mut combined = vec![0.0f32; FFT_SIZE];
    for &midi in notes {
        let sig = generate_piano_signal(midi, FFT_SIZE);
        for (i, s) in sig.iter().enumerate() {
            combined[i] += s;
        }
    }

    let magnitudes = compute_fft_magnitudes(&combined);
    let pitch_energy = dsp::compute_pitch_energies(&magnitudes, &templates);

    // Harmonic aliasing suppression + local-max ±2 semitones (same as inference)
    let abs_min_energy: f32 = 4.0;
    let mut filtered = pitch_energy;
    dsp::suppress_harmonic_aliasing(&mut filtered, 0.50);

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

    let surviving: Vec<u8> = (PIANO_LO..=PIANO_HI)
        .filter(|&p| filtered[p as usize] > abs_min_energy)
        .collect();

    println!("\n=== C MAJOR CHORD: C4 (60) + E4 (64) + G4 (67) ===");
    println!("Surviving pitches: {:?}", surviving);

    for &n in notes {
        assert!(surviving.contains(&n), "MIDI {} should survive", n);
    }
    assert!(surviving.len() <= 6, "Too many pitches: {:?}", surviving);

    println!("PASS: {} pitches, all 3 targets present", surviving.len());
}

#[test]
fn diagnose_harmonic_template_coverage() {
    let templates = dsp::build_harmonic_templates(SR, FFT_SIZE);

    println!("\n=== HARMONIC TEMPLATE COVERAGE ===");

    let mut covered = 0;
    let total = (PIANO_HI - PIANO_LO + 1) as usize;

    for midi in PIANO_LO..=PIANO_HI {
        if let Some(ref tmpl) = templates[midi as usize] {
            covered += 1;
            let hz = midi_to_hz(midi);
            println!(
                "  MIDI {:3} ({:7.1} Hz): {} harmonics, bins = {:?}",
                midi,
                hz,
                tmpl.bins.len(),
                tmpl.bins
            );
        }
    }

    println!(
        "\nCovered: {covered}/{total} pitches ({:.0}%)",
        covered as f32 / total as f32 * 100.0
    );

    // All 88 piano keys should have at least one harmonic template
    assert_eq!(
        covered, total,
        "All piano pitches should have harmonic templates"
    );
}

/// Test that the harmonic sieve correctly separates octaves.
/// A4 (440 Hz) and A5 (880 Hz) should NOT confuse each other.
#[test]
fn diagnose_octave_separation() {
    let templates = dsp::build_harmonic_templates(SR, FFT_SIZE);

    // Play A4 only
    let target: u8 = 69; // A4
    let octave_up: u8 = 81; // A5
    let signal = generate_piano_signal(target, FFT_SIZE);
    let magnitudes = compute_fft_magnitudes(&signal);
    let pitch_energy = dsp::compute_pitch_energies(&magnitudes, &templates);

    let target_e = pitch_energy[target as usize];
    let octave_e = pitch_energy[octave_up as usize];

    println!("\n=== OCTAVE SEPARATION: A4 vs A5 ===");
    println!("A4 (MIDI 69) energy: {:.2}", target_e);
    println!("A5 (MIDI 81) energy: {:.2}", octave_e);
    println!("Ratio A5/A4: {:.4}", octave_e / target_e);

    // The target should have significantly more energy than the octave
    assert!(
        target_e > octave_e * 1.5,
        "A4 energy ({:.2}) should be > 1.5x A5 energy ({:.2})",
        target_e,
        octave_e
    );

    println!(
        "PASS: A4 is stronger than A5 by factor {:.2}",
        target_e / octave_e
    );
}
