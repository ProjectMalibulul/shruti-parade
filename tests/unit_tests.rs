// ==========================================================================
// Shruti Parade — Unit Tests
// ==========================================================================

// We test the non-GPU modules (timing, events, config, dsp helpers,
// particles, and render utilities) without requiring a display or audio
// device.

// ---------------------------------------------------------------------------
// Timing (AudioClock)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod timing_tests {
    use shruti_parade::timing::AudioClock;
    use std::sync::Arc;

    #[test]
    fn new_clock_starts_at_zero() {
        let clock = AudioClock::new(48000);
        assert_eq!(clock.now_samples(), 0);
        assert!((clock.now_seconds() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn advance_increments_samples() {
        let clock = AudioClock::new(48000);
        clock.advance(48000);
        assert_eq!(clock.now_samples(), 48000);
    }

    #[test]
    fn now_seconds_accuracy() {
        let clock = AudioClock::new(48000);
        clock.advance(48000);
        let t = clock.now_seconds();
        assert!((t - 1.0).abs() < 1e-9, "Expected ~1.0s, got {t}");
    }

    #[test]
    fn sample_to_seconds() {
        let clock = AudioClock::new(44100);
        let t = clock.sample_to_seconds(44100);
        assert!((t - 1.0).abs() < 1e-9);
    }

    #[test]
    fn sample_rate_accessor() {
        let clock = AudioClock::new(96000);
        assert!((clock.sample_rate() - 96000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn advance_is_cumulative() {
        let clock = AudioClock::new(48000);
        clock.advance(1000);
        clock.advance(2000);
        clock.advance(3000);
        assert_eq!(clock.now_samples(), 6000);
    }

    #[test]
    fn concurrent_advance_and_read() {
        let clock = Arc::new(AudioClock::new(48000));
        let c2 = clock.clone();

        let writer = std::thread::spawn(move || {
            for _ in 0..10_000 {
                c2.advance(1);
            }
        });

        // Reader — just ensure no panic / data race.
        for _ in 0..10_000 {
            let _ = clock.now_seconds();
        }

        writer.join().unwrap();
        assert_eq!(clock.now_samples(), 10_000);
    }
}

// ---------------------------------------------------------------------------
// Events (struct layout / Pod checks)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod events_tests {
    use shruti_parade::events::*;

    #[test]
    fn note_instance_size_is_48_bytes() {
        assert_eq!(
            std::mem::size_of::<NoteInstance>(),
            48,
            "NoteInstance must be 48 bytes for GPU alignment"
        );
    }

    #[test]
    fn key_instance_size_is_48_bytes() {
        assert_eq!(
            std::mem::size_of::<KeyInstance>(),
            48,
            "KeyInstance must be 48 bytes for GPU alignment"
        );
    }

    #[test]
    fn particle_instance_size_is_32_bytes() {
        assert_eq!(
            std::mem::size_of::<ParticleInstance>(),
            32,
            "ParticleInstance must be 32 bytes"
        );
    }

    #[test]
    fn frame_uniforms_size_is_16_bytes() {
        assert_eq!(
            std::mem::size_of::<FrameUniforms>(),
            16,
            "FrameUniforms must be 16 bytes"
        );
    }

    #[test]
    fn note_event_kind_equality() {
        assert_eq!(NoteEventKind::NoteOn, NoteEventKind::NoteOn);
        assert_ne!(NoteEventKind::NoteOn, NoteEventKind::NoteOff);
    }

    #[test]
    fn visual_note_open_end() {
        let vn = VisualNote {
            pitch: 60,
            velocity: 100,
            start_time: 0.0,
            end_time: None,
        };
        assert!(vn.end_time.is_none());
    }

    #[test]
    fn note_instance_bytemuck_roundtrip() {
        let inst = NoteInstance {
            position: [0.1, 0.2],
            size: [0.3, 0.4],
            color: [1.0, 0.0, 0.0, 1.0],
            border_radius: 0.1,
            glow_intensity: 0.5,
            _pad: [0.0; 2],
        };
        let bytes = bytemuck::bytes_of(&inst);
        let back: &NoteInstance = bytemuck::from_bytes(bytes);
        assert!((back.position[0] - 0.1).abs() < f32::EPSILON);
        assert!((back.glow_intensity - 0.5).abs() < f32::EPSILON);
    }
}

// ---------------------------------------------------------------------------
// Config (Default sanity)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod config_tests {
    use shruti_parade::config::EngineConfig;

    #[test]
    fn default_config_values() {
        let cfg = EngineConfig::default();
        assert_eq!(cfg.audio.sample_rate, 48000);
        assert_eq!(cfg.dsp.fft_size, 4096);
        assert_eq!(cfg.dsp.n_mels, 229);
        assert!(cfg.render.bloom_enabled);
        assert!(cfg.render.particles_enabled);
        assert_eq!(cfg.render.width, 1280);
        assert_eq!(cfg.render.height, 720);
    }

    #[test]
    fn fft_size_is_power_of_two() {
        let cfg = EngineConfig::default();
        assert!(cfg.dsp.fft_size.is_power_of_two());
    }

    #[test]
    fn hop_divides_fft() {
        let cfg = EngineConfig::default();
        assert_eq!(cfg.dsp.fft_size % cfg.dsp.hop_size, 0);
    }
}

// ---------------------------------------------------------------------------
// Particles
// ---------------------------------------------------------------------------

#[cfg(test)]
mod particle_tests {
    use shruti_parade::particles::ParticleSystem;

    #[test]
    fn new_system_is_empty() {
        let sys = ParticleSystem::new();
        assert_eq!(sys.count(), 0);
        assert!(sys.instances().is_empty());
    }

    #[test]
    fn spawn_adds_particles() {
        let mut sys = ParticleSystem::new();
        sys.spawn([0.0, 0.0], [1.0, 1.0, 1.0, 1.0], 10);
        assert_eq!(sys.count(), 10);
    }

    #[test]
    fn tick_reduces_lifetime() {
        let mut sys = ParticleSystem::new();
        sys.spawn([0.0, 0.0], [1.0, 1.0, 1.0, 1.0], 5);
        // Tick long enough to kill all particles (life decreases at dt*1.5)
        sys.tick(1.0);
        // After 1.0s with rate 1.5, life = 1.0 - 1.5 = -0.5 → culled
        assert_eq!(sys.count(), 0, "All particles should be culled after 1s");
    }

    #[test]
    fn tick_preserves_live_particles() {
        let mut sys = ParticleSystem::new();
        sys.spawn([0.0, 0.0], [1.0, 1.0, 1.0, 1.0], 5);
        // Small tick — particles should survive
        sys.tick(0.01);
        assert_eq!(sys.count(), 5);
    }

    #[test]
    fn instances_match_count() {
        let mut sys = ParticleSystem::new();
        sys.spawn([0.0, 0.0], [1.0, 0.0, 0.0, 1.0], 20);
        let insts = sys.instances();
        assert_eq!(insts.len(), sys.count());
    }

    #[test]
    fn spawn_respects_max_budget() {
        let mut sys = ParticleSystem::new();
        // Try to spawn more than MAX_PARTICLES (2048)
        sys.spawn([0.0, 0.0], [1.0, 1.0, 1.0, 1.0], 3000);
        assert!(sys.count() <= 2048, "Must not exceed MAX_PARTICLES");
    }

    #[test]
    fn gravity_pulls_particles_down() {
        let mut sys = ParticleSystem::new();
        sys.spawn([0.0, 0.5], [1.0, 1.0, 1.0, 1.0], 1);
        let y_before = sys.instances()[0].position[1];
        sys.tick(0.1);
        if sys.count() > 0 {
            let y_after = sys.instances()[0].position[1];
            // y should have changed (velocity + gravity)
            assert_ne!(y_before, y_after, "Particle should have moved");
        }
    }

    #[test]
    fn instance_alpha_decreases_with_life() {
        let mut sys = ParticleSystem::new();
        sys.spawn([0.0, 0.0], [1.0, 1.0, 1.0, 1.0], 1);
        let alpha_before = sys.instances()[0].color[3];
        sys.tick(0.1);
        if sys.count() > 0 {
            let alpha_after = sys.instances()[0].color[3];
            assert!(
                alpha_after < alpha_before,
                "Alpha should fade: {alpha_before} -> {alpha_after}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// DSP helpers (mel filterbank, hz_to_mel, CircBuf)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod dsp_tests {
    use shruti_parade::dsp::*;

    #[test]
    fn hz_to_mel_zero() {
        assert!((hz_to_mel(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn mel_to_hz_zero() {
        assert!((mel_to_hz(0.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn hz_mel_roundtrip() {
        for &hz in &[100.0, 440.0, 1000.0, 4000.0, 8000.0] {
            let mel = hz_to_mel(hz);
            let back = mel_to_hz(mel);
            assert!(
                (back - hz).abs() < 0.01,
                "Roundtrip failed for {hz}: got {back}"
            );
        }
    }

    #[test]
    fn filterbank_shape() {
        let fb = build_mel_filterbank(40, 2048, 48000.0, 30.0, 8000.0);
        assert_eq!(fb.len(), 40, "Should have 40 mel filters");
        for (i, filt) in fb.iter().enumerate() {
            assert_eq!(filt.len(), 2048 / 2 + 1, "Filter {i} should cover n_bins");
        }
    }

    #[test]
    fn filterbank_nonnegative() {
        let fb = build_mel_filterbank(40, 2048, 48000.0, 30.0, 8000.0);
        for filt in &fb {
            for &w in filt {
                assert!(w >= 0.0, "Mel filter weights must be non-negative");
            }
        }
    }

    #[test]
    fn filterbank_triangular_peaks_at_most_one() {
        let fb = build_mel_filterbank(40, 2048, 48000.0, 30.0, 8000.0);
        for filt in &fb {
            for &w in filt {
                assert!(w <= 1.0 + 1e-6, "Mel filter weight should be ≤ 1.0");
            }
        }
    }

    #[test]
    fn circ_buf_linearise() {
        let mut cb = CircBuf::new(4);
        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
            cb.push(v);
        }
        // After pushing 6 values into a size-4 buffer:
        // buf = [5.0, 6.0, 3.0, 4.0], pos = 2
        // linearise → [3.0, 4.0, 5.0, 6.0]
        let mut out = vec![0.0; 4];
        cb.linearise_into(&mut out);
        assert_eq!(out, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn circ_buf_fresh_is_zeroed() {
        let cb = CircBuf::new(8);
        let mut out = vec![-1.0; 8];
        cb.linearise_into(&mut out);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn circ_buf_single_push() {
        let mut cb = CircBuf::new(4);
        cb.push(42.0);
        let mut out = vec![0.0; 4];
        cb.linearise_into(&mut out);
        // pos moved to 1, so linearise: [0.0, 0.0, 0.0, 42.0]
        // Actually: buf = [42.0, 0.0, 0.0, 0.0], pos = 1
        // linearise: tail = buf[1..] = [0.0, 0.0, 0.0], head = buf[..1] = [42.0]
        // out = [0.0, 0.0, 0.0, 42.0]
        assert_eq!(out, vec![0.0, 0.0, 0.0, 42.0]);
    }
}

// ---------------------------------------------------------------------------
// DSP edge cases (NaN safety, boundary pitches, silent input, etc.)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod dsp_edge_case_tests {
    use shruti_parade::dsp::*;

    // --- NaN safety ---

    #[test]
    fn suppress_harmonic_aliasing_nan_no_panic() {
        let mut energies = [0.0f32; 128];
        energies[60] = f32::NAN;
        energies[64] = 100.0;
        energies[67] = 50.0;
        // Must not panic — previously used partial_cmp().unwrap()
        suppress_harmonic_aliasing(&mut energies, 0.50);
    }

    #[test]
    fn suppress_harmonic_aliasing_inf_no_panic() {
        let mut energies = [0.0f32; 128];
        energies[60] = f32::INFINITY;
        energies[64] = 100.0;
        suppress_harmonic_aliasing(&mut energies, 0.50);
        // INF pitch should remain (strongest)
        assert!(energies[60].is_infinite());
    }

    // --- Silent / all-zero input ---

    #[test]
    fn compute_pitch_energies_all_zero_magnitudes() {
        let templates = build_harmonic_templates(48000.0, 4096);
        let magnitudes = vec![0.0f32; 4096 / 2 + 1];
        let energies = compute_pitch_energies(&magnitudes, &templates);
        for e in energies.iter() {
            assert_eq!(*e, 0.0, "Zero magnitudes should yield zero energy");
        }
    }

    #[test]
    fn suppress_harmonic_aliasing_all_zero() {
        let mut energies = [0.0f32; 128];
        suppress_harmonic_aliasing(&mut energies, 0.50);
        for e in energies.iter() {
            assert_eq!(*e, 0.0);
        }
    }

    // --- Boundary pitch templates ---

    #[test]
    fn template_exists_for_a0() {
        let templates = build_harmonic_templates(48000.0, 4096);
        assert!(
            templates[21].is_some(),
            "MIDI 21 (A0, 27.5 Hz) should have a template"
        );
    }

    #[test]
    fn template_exists_for_c8() {
        let templates = build_harmonic_templates(48000.0, 4096);
        assert!(
            templates[108].is_some(),
            "MIDI 108 (C8, 4186 Hz) should have a template"
        );
    }

    #[test]
    fn template_none_outside_piano_range() {
        let templates = build_harmonic_templates(48000.0, 4096);
        assert!(templates[0].is_none());
        assert!(templates[20].is_none());
        assert!(templates[109].is_none());
        assert!(templates[127].is_none());
    }

    #[test]
    fn a0_template_has_multiple_harmonics() {
        let templates = build_harmonic_templates(48000.0, 4096);
        let tmpl = templates[21].as_ref().unwrap();
        // A0 at 27.5 Hz with 48kHz SR and 4096 FFT:
        // Harmonics: 27.5, 55, 82.5, 110, 137.5, 165, 192.5, 220
        // All well below Nyquist (24kHz), so all 8 harmonics should be present
        assert_eq!(
            tmpl.bins.len(),
            8,
            "A0 should have all 8 harmonics below Nyquist"
        );
    }

    #[test]
    fn c8_template_has_limited_harmonics() {
        let templates = build_harmonic_templates(48000.0, 4096);
        let tmpl = templates[108].as_ref().unwrap();
        // C8 at 4186 Hz: 2nd harmonic = 8372, ..., 5th = 20930, 6th = 25116 (>Nyquist*0.95)
        // Should have 5 harmonics
        assert!(
            tmpl.bins.len() >= 2 && tmpl.bins.len() <= 6,
            "C8 should have limited harmonics due to Nyquist, got {}",
            tmpl.bins.len()
        );
    }

    // --- midi_to_hz spot checks ---

    #[test]
    fn midi_to_hz_a4() {
        assert!((midi_to_hz(69) - 440.0).abs() < 0.01);
    }

    #[test]
    fn midi_to_hz_a0() {
        assert!(
            (midi_to_hz(21) - 27.5).abs() < 0.1,
            "A0 should be ~27.5 Hz, got {}",
            midi_to_hz(21)
        );
    }

    #[test]
    fn midi_to_hz_c8() {
        let hz = midi_to_hz(108);
        assert!(
            (hz - 4186.0).abs() < 1.0,
            "C8 should be ~4186 Hz, got {}",
            hz
        );
    }

    #[test]
    fn midi_to_hz_middle_c() {
        let hz = midi_to_hz(60);
        assert!(
            (hz - 261.63).abs() < 0.1,
            "Middle C should be ~261.63 Hz, got {}",
            hz
        );
    }

    // --- Single-bin spike ---

    #[test]
    fn single_bin_spike_activates_nearby_pitches_only() {
        let templates = build_harmonic_templates(48000.0, 4096);
        let n_bins = 4096 / 2 + 1;
        let mut magnitudes = vec![0.0f32; n_bins];

        // Spike the bin corresponding to A4's fundamental (440 Hz)
        let a4_bin = (440.0_f32 * 4096.0_f32 / 48000.0_f32).round() as usize;
        magnitudes[a4_bin] = 1000.0;

        let energies = compute_pitch_energies(&magnitudes, &templates);

        // MIDI 69 (A4) should have some energy
        assert!(
            energies[69] > 0.0,
            "A4 should have energy from its fundamental bin"
        );

        // Distant pitches should have zero energy
        assert_eq!(energies[21], 0.0, "A0 shouldn't be activated by A4's bin");
    }

    // --- Concurrent note suppression fairness ---

    #[test]
    fn equal_energy_concurrent_notes_both_survive() {
        let mut energies = [0.0f32; 128];
        // C4 and E4 — a major third, not harmonically related
        energies[60] = 100.0;
        energies[64] = 100.0;
        suppress_harmonic_aliasing(&mut energies, 0.50);
        assert!(
            energies[60] > 0.0,
            "C4 should survive with equal energy to E4"
        );
        assert!(
            energies[64] > 0.0,
            "E4 should survive with equal energy to C4"
        );
    }

    #[test]
    fn perfect_fifth_both_survive_when_strong() {
        let mut energies = [0.0f32; 128];
        // C4 and G4 — a perfect fifth
        energies[60] = 100.0;
        energies[67] = 80.0; // 80% of C4's energy, above 50% threshold
        suppress_harmonic_aliasing(&mut energies, 0.50);
        assert!(energies[60] > 0.0, "C4 should survive (stronger note)");
        assert!(
            energies[67] > 0.0,
            "G4 should survive — energy is above ratio threshold"
        );
    }

    #[test]
    fn weak_harmonic_ghost_suppressed() {
        let mut energies = [0.0f32; 128];
        // A4 strong, A5 weak (should be suppressed as harmonic ghost)
        energies[69] = 100.0; // A4
        energies[81] = 10.0; // A5 (octave above, 10% energy — below 50% threshold)
        suppress_harmonic_aliasing(&mut energies, 0.50);
        assert!(energies[69] > 0.0, "A4 should survive");
        assert_eq!(
            energies[81], 0.0,
            "A5 should be suppressed as harmonic ghost of A4"
        );
    }

    // --- Fundamental gate ---

    #[test]
    fn fundamental_gate_rejects_phantom_pitch() {
        // Construct magnitudes where a pitch's harmonic bins have energy
        // but the fundamental bin does NOT — should be rejected by fundamental gate.
        let templates = build_harmonic_templates(48000.0, 4096);
        let n_bins = 4096 / 2 + 1;
        let mut magnitudes = vec![0.0f32; n_bins];

        // A4 (69) fundamental bin
        let a4_f0_bin = (440.0_f32 * 4096.0_f32 / 48000.0_f32).round() as usize;

        // Put energy ONLY at A4's 2nd harmonic (880 Hz), not at fundamental
        let a4_h2_bin = (880.0_f32 * 4096.0_f32 / 48000.0_f32).round() as usize;
        magnitudes[a4_h2_bin] = 1000.0;
        // Make sure fundamental is empty
        magnitudes[a4_f0_bin] = 0.0;

        let energies = compute_pitch_energies(&magnitudes, &templates);

        // A4 should have very low or zero energy due to fundamental gate
        // (only 1 harmonic has energy, fundamental has none)
        // The fundamental gate requires fund_mag >= score * FUNDAMENTAL_GATE (0.10)
        // fund_mag = 0, so A4 should be rejected
        assert_eq!(
            energies[69], 0.0,
            "A4 should be rejected: no energy at fundamental"
        );
    }
}

// ---------------------------------------------------------------------------
// Render utilities (pitch_to_ndc_x, pitch_to_hue, hsv_to_rgb, piano keys)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod render_util_tests {
    use shruti_parade::render::*;

    #[test]
    fn piano_min_maps_to_first_key() {
        let x = pitch_to_ndc_x(21); // A0 = PIANO_MIN
                                    // Key-aligned: first white key center = -1.0 + 0.5 * (2.0/52.0)
        let expected = -1.0 + 0.5 * (2.0 / 52.0);
        assert!(
            (x - expected).abs() < 1e-5,
            "PIANO_MIN should map to first key center {expected}, got {x}"
        );
    }

    #[test]
    fn piano_max_maps_to_last_key() {
        let x = pitch_to_ndc_x(108); // C8 = PIANO_MAX
                                     // Key-aligned: white key index 51, center = -1.0 + 51.5 * (2.0/52.0)
        let expected = -1.0 + 51.5 * (2.0 / 52.0);
        assert!(
            (x - expected).abs() < 1e-5,
            "PIANO_MAX should map to last key center {expected}, got {x}"
        );
    }

    #[test]
    fn below_piano_min_clamps() {
        let x = pitch_to_ndc_x(0);
        // saturating_sub(21) = 0 → maps to -1.0
        assert!((x - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn hue_range_is_0_to_300() {
        let hue_min = pitch_to_hue(21);
        let hue_max = pitch_to_hue(108);
        assert!((hue_min - 0.0).abs() < 1e-6);
        assert!((hue_max - 300.0).abs() < 1e-6);
    }

    #[test]
    fn hsv_to_rgb_red() {
        let (r, g, b) = hsv_to_rgb(0.0, 1.0, 1.0);
        assert!((r - 1.0).abs() < 1e-6);
        assert!(g.abs() < 1e-6);
        assert!(b.abs() < 1e-6);
    }

    #[test]
    fn hsv_to_rgb_green() {
        let (r, g, b) = hsv_to_rgb(120.0, 1.0, 1.0);
        assert!(r.abs() < 1e-6);
        assert!((g - 1.0).abs() < 1e-6);
        assert!(b.abs() < 1e-6);
    }

    #[test]
    fn hsv_to_rgb_blue() {
        let (r, g, b) = hsv_to_rgb(240.0, 1.0, 1.0);
        assert!(r.abs() < 1e-6);
        assert!(g.abs() < 1e-6);
        assert!((b - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hsv_to_rgb_white() {
        let (r, g, b) = hsv_to_rgb(0.0, 0.0, 1.0);
        assert!((r - 1.0).abs() < 1e-6);
        assert!((g - 1.0).abs() < 1e-6);
        assert!((b - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hsv_to_rgb_black() {
        let (r, g, b) = hsv_to_rgb(0.0, 0.0, 0.0);
        assert!(r.abs() < 1e-6);
        assert!(g.abs() < 1e-6);
        assert!(b.abs() < 1e-6);
    }

    #[test]
    fn is_black_key_csharp() {
        assert!(is_black_key(61)); // C#4 (MIDI 61, 61%12=1)
    }

    #[test]
    fn is_black_key_c() {
        assert!(!is_black_key(60)); // C4 (MIDI 60, 60%12=0)
    }

    #[test]
    fn is_black_key_all_12_classes() {
        // black keys: 1(C#), 3(D#), 6(F#), 8(G#), 10(A#)
        for note in 0..12u8 {
            let expected = matches!(note, 1 | 3 | 6 | 8 | 10);
            assert_eq!(
                is_black_key(note + 60),
                expected,
                "Note class {note} (midi {}) mismatch",
                note + 60
            );
        }
    }

    #[test]
    fn build_piano_keys_count() {
        let keys = build_piano_keys();
        // 88 keys + 2 decorative elements (shadow bar + highlight strip)
        assert_eq!(
            keys.len(),
            90,
            "Expected 90 piano key instances, got {}",
            keys.len()
        );
    }

    #[test]
    fn white_keys_brighter_than_black() {
        let keys = build_piano_keys();
        // First 2 are decorative (shadow + highlight), then 52 white, then 36 black
        for key in &keys[2..54] {
            let luminance = key.color[0] * 0.3 + key.color[1] * 0.59 + key.color[2] * 0.11;
            assert!(luminance > 0.5, "White key should be bright");
        }
        for key in &keys[54..] {
            let luminance = key.color[0] * 0.3 + key.color[1] * 0.59 + key.color[2] * 0.11;
            assert!(luminance < 0.3, "Black key should be dark");
        }
    }
}
