use anyhow::{Context, Result};
use std::path::Path;
use tracing::info;

/// Read the sample rate from a WAV file without consuming it.
#[allow(dead_code)]
pub fn wav_sample_rate(path: &Path) -> Result<u32> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("Failed to open WAV file: {}", path.display()))?;
    Ok(reader.spec().sample_rate)
}

/// Load a WAV file and push its samples into an rtrb producer.
/// Returns the sample rate reported by the file.
///
/// This drives the audio ring at ~real-time by sleeping between chunks,
/// allowing the DSP pipeline to consume at the same pace as live audio.
///
/// If `playback_producer` is `Some`, samples are also pushed to a second
/// ring for audio output playback.
#[allow(dead_code)]
pub fn stream_wav_file(
    path: &Path,
    mut producer: rtrb::Producer<f32>,
    mut playback_producer: Option<rtrb::Producer<f32>>,
    clock: std::sync::Arc<crate::timing::AudioClock>,
    chunk_size: usize,
) -> Result<()> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("Failed to open WAV file: {}", path.display()))?;

    let spec = reader.spec();
    info!(
        "WAV: {} ch, {} Hz, {:?} {}bit",
        spec.channels, spec.sample_rate, spec.sample_format, spec.bits_per_sample
    );

    let sr = spec.sample_rate as f64;
    let sleep_per_chunk = std::time::Duration::from_secs_f64(chunk_size as f64 / sr);

    // Convert all samples to mono f32 regardless of source format.
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .filter_map(|s| s.ok())
            .collect(),
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Downmix to mono if stereo+.
    let mono: Vec<f32> = if spec.channels > 1 {
        let ch = spec.channels as usize;
        samples
            .chunks_exact(ch)
            .map(|frame| frame.iter().sum::<f32>() / ch as f32)
            .collect()
    } else {
        samples
    };

    info!(
        "WAV loaded: {} mono samples ({:.1}s)",
        mono.len(),
        mono.len() as f64 / sr
    );

    // Stream at ~real-time pace.
    for chunk in mono.chunks(chunk_size) {
        for &sample in chunk {
            // Spin-wait briefly if ring is full (shouldn't happen at real-time pace).
            while producer.push(sample).is_err() {
                std::hint::spin_loop();
            }
            // Also push to playback ring if present.
            if let Some(ref mut pb) = playback_producer {
                while pb.push(sample).is_err() {
                    std::hint::spin_loop();
                }
            }
        }
        clock.advance(chunk.len() as u64);
        std::thread::sleep(sleep_per_chunk);
    }

    info!("WAV playback complete");
    Ok(())
}
