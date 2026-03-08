use shruti_parade::config::{DspConfig, InferenceConfig};
use shruti_parade::dsp::{DspPipeline, PitchFrame};
use shruti_parade::events::{NoteEvent, NoteEventKind};
use shruti_parade::inference::InferenceEngine;

#[test]
fn onset_fires_on_impulse() {
    let sample_rate: u32 = 48000;
    let fft_size: usize = 4096;
    let hop_size: usize = 512;

    // Set up rtrb ring buffer for DSP input
    let (mut producer, consumer) = rtrb::RingBuffer::new(sample_rate as usize);

    // DSP → Inference channel
    let (pitch_tx, pitch_rx) = crossbeam_channel::unbounded::<PitchFrame>();

    // Inference → output channel
    let (event_tx, event_rx) = crossbeam_channel::unbounded::<NoteEvent>();

    let dsp_config = DspConfig {
        fft_size,
        bass_fft_size: 8192,
        hop_size,
        n_mels: 229,
        mel_fmin: 30.0,
        mel_fmax: 8000.0,
    };

    let inference_config = InferenceConfig {
        model_path: String::new(),
        context_frames: 32,
        overlap_frames: 8,
        onset_threshold: 0.5,
        frame_threshold: 0.3,
    };

    // Spawn inference thread
    let _inference_handle = std::thread::spawn(move || {
        let mut engine = InferenceEngine::new(inference_config, pitch_rx, event_tx);
        engine.run();
    });

    // Spawn DSP thread
    let _dsp_handle = std::thread::spawn(move || {
        let mut pipeline = DspPipeline::new(dsp_config, consumer, pitch_tx, sample_rate);
        pipeline.run();
    });

    // Feed silence first (warmup: at least 30 hops worth of samples)
    let warmup_samples = hop_size * 35;
    for _ in 0..warmup_samples {
        while producer.push(0.0).is_err() {
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
    }

    // Small gap to let warmup frames process
    std::thread::sleep(std::time::Duration::from_millis(50));

    // Feed a broadband noise burst (simulates an impulsive onset)
    // Use a deterministic pseudo-random sequence
    let burst_samples = fft_size * 2;
    let mut seed: u32 = 12345;
    for _ in 0..burst_samples {
        // Simple LCG PRNG
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let noise = (seed as f32 / u32::MAX as f32) * 2.0 - 1.0;
        while producer.push(noise * 0.8).is_err() {
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
    }

    // Feed trailing silence so offset can fire
    let tail_samples = hop_size * 20;
    for _ in 0..tail_samples {
        while producer.push(0.0).is_err() {
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
    }

    // Wait for processing, with a timeout of 100ms wall time
    std::thread::sleep(std::time::Duration::from_millis(100));

    // Drop producer to close the ring buffer, which will eventually
    // cause the DSP thread to spin on empty ring. We need to also
    // drop the pitch sender (held by DSP thread) to close inference.
    drop(producer);

    // Give threads time to finish
    std::thread::sleep(std::time::Duration::from_millis(200));

    // Collect events (non-blocking)
    let events: Vec<NoteEvent> = event_rx.try_iter().collect();

    let has_note_on = events
        .iter()
        .any(|e| e.kind == NoteEventKind::NoteOn);

    assert!(
        has_note_on,
        "Expected at least one NoteOn event from broadband noise burst, got {} events: {:?}",
        events.len(),
        events,
    );

    // Clean up: the threads may still be blocked. We can't easily join them
    // because the DSP thread loops forever on the ring buffer. That's OK for
    // a test — the process will exit.
}
