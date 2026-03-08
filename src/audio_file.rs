//! Generic audio file decoder using symphonia.
//! Supports WAV, MP3, FLAC, OGG Vorbis, and any other format symphonia handles.

use std::path::Path;

use anyhow::{Context, Result};
use crossbeam_channel::Receiver;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use tracing::info;

use crate::transport::{TransportCommand, TransportState};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Peek the sample rate from an audio file header without decoding.
pub fn audio_file_sample_rate(path: &Path) -> Result<u32> {
    let file =
        std::fs::File::open(path).with_context(|| format!("Cannot open: {}", path.display()))?;
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
    let file =
        std::fs::File::open(path).with_context(|| format!("Cannot open: {}", path.display()))?;
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
    let n_channels = track.codec_params.channels.map(|c| c.count()).unwrap_or(1);

    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut all_samples: Vec<f32> = Vec::new();

    while let Ok(packet) = format.next_packet() {
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
    transport_rx: Receiver<TransportCommand>,
    transport_state: std::sync::Arc<TransportState>,
) -> Result<()> {
    let (mono, _sr) = decode_audio_file(path)?;
    transport_state.set_total_samples(mono.len() as u64);

    let mut cursor: usize = 0;
    let mut paused = transport_state.is_paused();

    // ---- Pre-fill rings with initial audio for output headroom ----
    let prefill = (chunk_size * 4).min(mono.len());
    for &sample in &mono[cursor..prefill] {
        let _ = producer.push(sample);
        if let Some(ref mut pb) = playback_producer {
            let _ = pb.push(sample);
        }
    }
    cursor = prefill;
    clock.advance(prefill as u64);

    while cursor < mono.len() {
        while let Ok(cmd) = transport_rx.try_recv() {
            match cmd {
                TransportCommand::Play => {
                    paused = false;
                    transport_state.set_paused(false);
                }
                TransportCommand::Pause => {
                    paused = true;
                    transport_state.set_paused(true);
                }
                TransportCommand::SeekSamples(sample) => {
                    let target = sample.min(mono.len() as u64) as usize;
                    cursor = target;
                    clock.set_samples(target as u64);
                }
            }
        }

        if paused {
            std::thread::sleep(std::time::Duration::from_millis(10));
            continue;
        }

        // ---- Ring-occupancy pacing: push only when there's room ----
        let dsp_slots = producer.slots();
        let pb_slots = playback_producer
            .as_ref()
            .map(|p| p.slots())
            .unwrap_or(usize::MAX);
        let available = dsp_slots.min(pb_slots);

        if available >= chunk_size {
            let chunk_end = (cursor + chunk_size).min(mono.len());
            let chunk = &mono[cursor..chunk_end];

            for &sample in chunk {
                let _ = producer.push(sample);
                if let Some(ref mut pb) = playback_producer {
                    let _ = pb.push(sample);
                }
            }

            clock.advance(chunk.len() as u64);
            cursor = chunk_end;
        } else {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    info!("Audio file playback complete");
    Ok(())
}
