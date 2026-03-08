use std::sync::Arc;

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream, StreamConfig};
use tracing::{error, info};

use crate::config::AudioConfig;
use crate::timing::AudioClock;

/// Owns the `cpal` input stream. Dropping this stops capture.
pub struct AudioCapture {
    _stream: Stream,
}

/// Owns the `cpal` output stream for WAV playback. Dropping this stops playback.
pub struct AudioPlayback {
    _stream: Stream,
}

impl AudioPlayback {
    /// Start playing audio from the provided SPSC ring consumer.
    ///
    /// The audio callback is **real-time safe**: it only pops samples via
    /// wait-free `rtrb::Consumer::pop`, outputting silence when the ring is empty.
    /// `sample_rate` should match the WAV file's native rate so playback
    /// runs at the correct speed without resampling.
    pub fn start(
        sample_rate: u32,
        buffer_frames: usize,
        mut consumer: rtrb::Consumer<f32>,
    ) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .context("No audio output device found")?;

        info!("Audio output device: {}", device.name().unwrap_or_default());

        let stream_config = StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let mut last_sample = 0.0f32;
        let mut underrun_count = 0u64;

        let stream = device.build_output_stream(
            &stream_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                for sample in data.iter_mut() {
                    match consumer.pop() {
                        Ok(s) => {
                            last_sample = s;
                            *sample = s;
                        }
                        Err(_) => {
                            // Sample-hold fallback: avoids click from abrupt drop to 0
                            *sample = last_sample;
                            underrun_count += 1;
                            if underrun_count & (underrun_count - 1) == 0 {
                                eprintln!("[audio] ring underrun (×{underrun_count})");
                            }
                        }
                    }
                }
            },
            |err| eprintln!("[audio] output stream error: {err}"),
            None,
        )?;

        stream.play()?;
        info!(
            "Audio playback live — {}Hz, {} frame buffer",
            sample_rate, buffer_frames
        );

        Ok(Self { _stream: stream })
    }
}

impl AudioCapture {
    /// Start capturing audio into the provided SPSC ring producer.
    ///
    /// The audio callback is **real-time safe**: no heap alloc, no locks, no
    /// syscalls — it only pushes individual samples via wait-free `rtrb::Producer::push`.
    pub fn start(
        config: &AudioConfig,
        clock: Arc<AudioClock>,
        mut producer: rtrb::Producer<f32>,
    ) -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .context("No audio input device found")?;

        info!("Audio input device: {}", device.name().unwrap_or_default());

        let supported = device.default_input_config()?;
        info!("Supported config: {:?}", supported);

        let stream_config = StreamConfig {
            channels: 1, // mono
            sample_rate: cpal::SampleRate(config.sample_rate),
            buffer_size: cpal::BufferSize::Fixed(config.buffer_frames as u32),
        };

        let stream = match supported.sample_format() {
            SampleFormat::F32 => device.build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    // RT-safe hot path: push each sample, drop if ring full.
                    for &sample in data {
                        let _ = producer.push(sample);
                    }
                    clock.advance(data.len() as u64);
                },
                |err| error!("Audio stream error: {err}"),
                None,
            )?,
            fmt => anyhow::bail!("Unsupported sample format: {fmt:?}"),
        };

        stream.play()?;
        info!(
            "Audio capture live — {}Hz, {} frame buffer",
            config.sample_rate, config.buffer_frames
        );

        Ok(Self { _stream: stream })
    }
}
