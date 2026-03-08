use serde::Deserialize;

/// Top-level engine configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct EngineConfig {
    pub audio: AudioConfig,
    pub dsp: DspConfig,
    pub inference: InferenceConfig,
    pub render: RenderConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub buffer_frames: usize,
    pub ring_capacity: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DspConfig {
    pub fft_size: usize,
    pub bass_fft_size: usize,
    pub hop_size: usize,
    #[allow(dead_code)]
    pub n_mels: usize,
    #[allow(dead_code)]
    pub mel_fmin: f32,
    #[allow(dead_code)]
    pub mel_fmax: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InferenceConfig {
    #[allow(dead_code)]
    pub model_path: String,
    #[allow(dead_code)]
    pub context_frames: usize,
    #[allow(dead_code)]
    pub overlap_frames: usize,
    pub onset_threshold: f32,
    pub frame_threshold: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RenderConfig {
    pub width: u32,
    pub height: u32,
    #[allow(dead_code)]
    pub note_fall_speed: f32,
    pub bloom_enabled: bool,
    pub particles_enabled: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            audio: AudioConfig {
                sample_rate: 48000,
                buffer_frames: 512,
                ring_capacity: 48000, // 1s at 48kHz
            },
            dsp: DspConfig {
                fft_size: 4096,
                bass_fft_size: 8192,
                hop_size: 512,
                n_mels: 229,
                mel_fmin: 30.0,
                mel_fmax: 8000.0,
            },
            inference: InferenceConfig {
                model_path: String::from("models/piano_transcription.onnx"),
                context_frames: 32,
                overlap_frames: 8,
                onset_threshold: 0.5,
                frame_threshold: 0.3,
            },
            render: RenderConfig {
                width: 1280,
                height: 720,
                note_fall_speed: 300.0,
                bloom_enabled: true,
                particles_enabled: true,
            },
        }
    }
}
