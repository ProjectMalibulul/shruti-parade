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
/// Falls back to a drain thread when no audio device is available (e.g. CI).
pub struct AudioPlayback {
    _stream: Option<Stream>,
    _drain: Option<std::thread::JoinHandle<()>>,
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
        let device = match host.default_output_device() {
            Some(d) => d,
            None => {
                info!("No audio output device — running headless (no playback)");
                let drain = std::thread::Builder::new()
                    .name("audio-drain".into())
                    .spawn(move || {
                        // Drain ring so producers never block
                        loop {
                            match consumer.pop() {
                                Ok(_) => {}
                                Err(_) => std::thread::sleep(std::time::Duration::from_millis(1)),
                            }
                        }
                    })
                    .ok();
                return Ok(Self {
                    _stream: None,
                    _drain: drain,
                });
            }
        };

        info!("Audio output device: {}", device.name().unwrap_or_default());

        let stream_config = StreamConfig {
            channels: 2, // stereo — mono samples are duplicated to L+R in the callback
            sample_rate: cpal::SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        let mut last_sample = 0.0f32;
        let mut underrun_count = 0u64;

        let stream = device.build_output_stream(
            &stream_config,
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                // Iterate one stereo frame (2 samples) at a time, popping one
                // mono sample from the ring per frame and duplicating to L+R.
                for frame in data.chunks_mut(2) {
                    match consumer.pop() {
                        Ok(s) => {
                            last_sample = s;
                            for ch in frame.iter_mut() {
                                *ch = s;
                            }
                        }
                        Err(_) => {
                            // Sample-hold fallback: avoids click from abrupt drop to 0
                            for ch in frame.iter_mut() {
                                *ch = last_sample;
                            }
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

        Ok(Self {
            _stream: Some(stream),
            _drain: None,
        })
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
