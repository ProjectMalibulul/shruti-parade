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
    use std::sync::Arc;
    use shruti_parade::timing::AudioClock;

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
        assert_eq!(cfg.dsp.fft_size, 2048);
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
            assert_eq!(
                filt.len(),
                2048 / 2 + 1,
                "Filter {i} should cover n_bins"
            );
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
// Render utilities (pitch_to_ndc_x, pitch_to_hue, hsv_to_rgb, piano keys)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod render_util_tests {
    use shruti_parade::render::*;

    #[test]
    fn piano_min_maps_to_minus_one() {
        let x = pitch_to_ndc_x(21); // A0 = PIANO_MIN
        assert!((x - (-1.0)).abs() < 1e-6, "PIANO_MIN should map to -1.0, got {x}");
    }

    #[test]
    fn piano_max_maps_to_near_plus_one() {
        let x = pitch_to_ndc_x(108); // C8 = PIANO_MAX
        // Should be close to +1.0 (exactly 2.0 * 87.0 / 87.0 - 1.0 = 1.0)
        assert!((x - 1.0).abs() < 1e-6, "PIANO_MAX should map to +1.0, got {x}");
    }

    #[test]
    fn below_piano_min_clamps() {
        let x = pitch_to_ndc_x(0);
        // saturating_sub(21) = 0 → maps to -1.0
        assert!((x - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn hue_range_is_0_to_360() {
        let hue_min = pitch_to_hue(21);
        let hue_max = pitch_to_hue(108);
        assert!((hue_min - 0.0).abs() < 1e-6);
        assert!((hue_max - 360.0).abs() < 1e-6);
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
        // 88 keys total: 52 white + 36 black
        assert_eq!(keys.len(), 88, "Expected 88 piano keys, got {}", keys.len());
    }

    #[test]
    fn white_keys_brighter_than_black() {
        let keys = build_piano_keys();
        // First 52 keys are white, rest are black
        for key in &keys[..52] {
            let luminance = key.color[0] * 0.3 + key.color[1] * 0.59 + key.color[2] * 0.11;
            assert!(luminance > 0.5, "White key should be bright");
        }
        for key in &keys[52..] {
            let luminance = key.color[0] * 0.3 + key.color[1] * 0.59 + key.color[2] * 0.11;
            assert!(luminance < 0.3, "Black key should be dark");
        }
    }
}
