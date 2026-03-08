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

/// Decode an entire audio file to interleaved f32 samples.
/// Returns `(interleaved_samples, sample_rate, n_channels)`.
pub fn decode_audio_file(path: &Path) -> Result<(Vec<f32>, u32, usize)> {
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

    let total_frames = all_samples.len() / n_channels;

    info!(
        "Audio decoded: {} ch, {} Hz, {} frames ({:.1}s)",
        n_channels,
        sample_rate,
        total_frames,
        total_frames as f64 / sample_rate as f64
    );

    Ok((all_samples, sample_rate, n_channels))
}

// ---------------------------------------------------------------------------
// Real-time streaming
// ---------------------------------------------------------------------------

/// Decode an audio file and stream it at real-time pace into the DSP and
/// playback ring buffers.
///
/// The DSP ring receives **mono** samples (downmixed on the fly).
/// The playback ring receives **stereo interleaved** samples (preserving the
/// original stereo field, or duplicating mono to L+R).
/// The two rings are fed **independently** so that a slow DSP consumer cannot
/// starve the playback ring.
pub fn stream_audio_file(
    path: &Path,
    mut producer: rtrb::Producer<f32>,
    mut playback_producer: Option<rtrb::Producer<f32>>,
    clock: std::sync::Arc<crate::timing::AudioClock>,
    chunk_size: usize,
    transport_rx: Receiver<TransportCommand>,
    transport_state: std::sync::Arc<TransportState>,
) -> Result<()> {
    let (samples, _sr, n_channels) = decode_audio_file(path)?;
    let total_frames = samples.len() / n_channels;
    transport_state.set_total_samples(total_frames as u64);

    // Pre-compute mono for the DSP ring (pitch detection needs mono).
    let mono: Vec<f32> = if n_channels > 1 {
        samples
            .chunks(n_channels)
            .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
            .collect()
    } else {
        samples.clone()
    };

    let mut paused = transport_state.is_paused();

    // ---- Pre-fill rings with initial audio for output headroom ----
    let prefill = (chunk_size * 4).min(total_frames);
    for (m, frame) in mono[..prefill].iter().zip(samples.chunks(n_channels)) {
        let _ = producer.push(*m);
        if let Some(ref mut pb) = playback_producer {
            let _ = pb.push(frame[0]);
            let _ = pb.push(if n_channels >= 2 { frame[1] } else { frame[0] });
        }
    }
    let mut dsp_cursor: usize = prefill;
    let mut pb_cursor: usize = prefill;
    let mut clock_pos: usize = prefill;
    clock.advance(prefill as u64);

    while dsp_cursor < total_frames || pb_cursor < total_frames {
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
                    let target = sample.min(total_frames as u64) as usize;
                    dsp_cursor = target;
                    pb_cursor = target;
                    clock_pos = target;
                    clock.set_samples(target as u64);
                }
            }
        }

        if paused {
            std::thread::sleep(std::time::Duration::from_millis(10));
            continue;
        }

        let mut did_work = false;

        // ---- Feed DSP ring (mono) ----
        if dsp_cursor < total_frames {
            let dsp_slots = producer.slots();
            if dsp_slots >= chunk_size {
                let end = (dsp_cursor + chunk_size).min(total_frames);
                for &s in &mono[dsp_cursor..end] {
                    let _ = producer.push(s);
                }
                dsp_cursor = end;
                did_work = true;
            }
        }

        // ---- Feed playback ring (stereo interleaved, independent of DSP) ----
        if pb_cursor < total_frames {
            if let Some(ref mut pb) = playback_producer {
                let pb_slots = pb.slots();
                let needed = chunk_size * 2; // stereo: 2 samples per frame
                if pb_slots >= needed {
                    let end = (pb_cursor + chunk_size).min(total_frames);
                    for frame in
                        samples[pb_cursor * n_channels..end * n_channels].chunks(n_channels)
                    {
                        let _ = pb.push(frame[0]);
                        let _ = pb.push(if n_channels >= 2 { frame[1] } else { frame[0] });
                    }
                    pb_cursor = end;
                    did_work = true;
                }
            } else {
                pb_cursor = total_frames;
            }
        }

        // ---- Advance clock to the slower consumer's position ----
        let new_pos = dsp_cursor.min(pb_cursor);
        if new_pos > clock_pos {
            clock.advance((new_pos - clock_pos) as u64);
            clock_pos = new_pos;
        }

        if !did_work {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    info!("Audio file playback complete");
    Ok(())
}
