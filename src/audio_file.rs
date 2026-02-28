//! Generic audio file decoder using symphonia.
//! Supports WAV, MP3, FLAC, OGG Vorbis, and any other format symphonia handles.

use std::path::Path;

use anyhow::{Context, Result};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use tracing::info;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Peek the sample rate from an audio file header without decoding.
pub fn audio_file_sample_rate(path: &Path) -> Result<u32> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open: {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let track = probed
        .format
        .default_track()
        .context("No audio track found")?;
    Ok(track.codec_params.sample_rate.unwrap_or(44100))
}

// ---------------------------------------------------------------------------
// Full decode
// ---------------------------------------------------------------------------

/// Decode an entire audio file to mono f32 samples.
/// Returns `(samples, sample_rate)`.
pub fn decode_audio_file(path: &Path) -> Result<(Vec<f32>, u32)> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open: {}", path.display()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .context("No audio track found")?
        .clone();

    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);
    let n_channels = track
        .codec_params
        .channels
        .map(|c| c.count())
        .unwrap_or(1);

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())?;

    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            // End of stream (various formats signal this differently)
            Err(_) => break,
        };

        if packet.track_id() != track.id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                let duration = decoded.capacity();
                if duration == 0 {
                    continue;
                }

                let mut sbuf = SampleBuffer::<f32>::new(duration as u64, spec);
                sbuf.copy_interleaved_ref(decoded);
                all_samples.extend_from_slice(sbuf.samples());
            }
            // Skip corrupt frames rather than aborting
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(_) => break,
        }
    }

    // Downmix to mono
    let mono: Vec<f32> = if n_channels > 1 {
        all_samples
            .chunks(n_channels)
            .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
            .collect()
    } else {
        all_samples
    };

    info!(
        "Audio decoded: {} ch, {} Hz, {} mono samples ({:.1}s)",
        n_channels,
        sample_rate,
        mono.len(),
        mono.len() as f64 / sample_rate as f64
    );

    Ok((mono, sample_rate))
}

// ---------------------------------------------------------------------------
// Real-time streaming
// ---------------------------------------------------------------------------

/// Decode an audio file and stream it at real-time pace into the DSP and
/// playback ring buffers.
pub fn stream_audio_file(
    path: &Path,
    mut producer: rtrb::Producer<f32>,
    mut playback_producer: Option<rtrb::Producer<f32>>,
    clock: std::sync::Arc<crate::timing::AudioClock>,
    chunk_size: usize,
) -> Result<()> {
    let (mono, sr) = decode_audio_file(path)?;
    let sleep_per_chunk = std::time::Duration::from_secs_f64(chunk_size as f64 / sr as f64);

    for chunk in mono.chunks(chunk_size) {
        for &sample in chunk {
            while producer.push(sample).is_err() {
                std::hint::spin_loop();
            }
            if let Some(ref mut pb) = playback_producer {
                while pb.push(sample).is_err() {
                    std::hint::spin_loop();
                }
            }
        }
        clock.advance(chunk.len() as u64);
        std::thread::sleep(sleep_per_chunk);
    }

    info!("Audio file playback complete");
    Ok(())
}
