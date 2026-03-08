use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender};
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use crate::config::RenderConfig;
use crate::events::{
    FrameUniforms, KeyInstance, NoteEvent, NoteEventKind, NoteInstance, ParticleInstance,
    VisualNote,
};
use crate::particles::ParticleSystem;
use crate::timing::AudioClock;
use crate::transport::{TransportCommand, TransportState};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_NOTE_INSTANCES: usize = 4096;
const MAX_PARTICLE_INSTANCES: usize = 2048;
const MAX_UI_INSTANCES: usize = 256;
const MAX_KEY_INSTANCES: usize = 384; // 88 keys + pedal + heat bar + key names

const HIT_LINE_Y: f32 = -0.8; // NDC y where notes "land"
const SCROLL_SPEED: f32 = 0.5; // NDC per second

const PIANO_MIN: u8 = 21; // A0
const PIANO_MAX: u8 = 108; // C8
const PIANO_RANGE: f32 = (PIANO_MAX - PIANO_MIN) as f32;

const NOTE_CULL_SECONDS: f64 = 12.0;
const PARTICLE_BURST: usize = 12;

const PLAYBAR_Y: f32 = 0.86;
const PLAYBAR_WIDTH: f32 = 1.4;
const PLAYBAR_HEIGHT: f32 = 0.04;
const PLAYBAR_PADDING: f32 = 0.02;
const PLAYBAR_HANDLE_WIDTH: f32 = 0.02;
const PLAY_BUTTON_SIZE: f32 = 0.07;
const PLAY_BUTTON_X: f32 = -0.85;
const PLAYBAR_SEEK_SECONDS: f64 = 5.0;

const FILE_BUTTON_X: f32 = 0.85;
const FILE_BUTTON_Y: f32 = PLAYBAR_Y;
const FILE_BUTTON_SIZE: f32 = 0.07;

// ---------------------------------------------------------------------------
// Vertex type and quad geometry
// ---------------------------------------------------------------------------

/// Per-vertex data for the unit quad (position only, maps to @location(0) in WGSL).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
}

/// Unit quad as two triangles (TriangleList). Covers [-1,1] in both axes.
/// Instanced pipelines scale and translate this per-instance.
const QUAD_VERTS: [Vertex; 6] = [
    Vertex {
        position: [-1.0, -1.0],
    },
    Vertex {
        position: [1.0, -1.0],
    },
    Vertex {
        position: [1.0, 1.0],
    },
    Vertex {
        position: [-1.0, -1.0],
    },
    Vertex {
        position: [1.0, 1.0],
    },
    Vertex {
        position: [-1.0, 1.0],
    },
];

// ---------------------------------------------------------------------------
// Playbar layout
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Rect {
    center: [f32; 2],
    size: [f32; 2],
}

impl Rect {
    fn contains(&self, point: [f32; 2]) -> bool {
        let half_w = self.size[0] * 0.5;
        let half_h = self.size[1] * 0.5;
        point[0] >= self.center[0] - half_w
            && point[0] <= self.center[0] + half_w
            && point[1] >= self.center[1] - half_h
            && point[1] <= self.center[1] + half_h
    }

    fn left(&self) -> f32 {
        self.center[0] - self.size[0] * 0.5
    }
}

#[derive(Clone, Copy)]
struct PlaybarLayout {
    bar: Rect,
    button: Rect,
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn run_render(
    config: RenderConfig,
    event_rx: Receiver<NoteEvent>,
    clock: Arc<AudioClock>,
    transport_tx: Sender<TransportCommand>,
    transport_state: Arc<TransportState>,
    pending_file: Arc<Mutex<Option<PathBuf>>>,
) -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    let mut app = App {
        config,
        event_rx,
        clock,
        transport_tx,
        transport_state,
        window: None,
        gpu: None,
        visual_notes: Vec::new(),
        particles: ParticleSystem::new(),
        last_frame: Instant::now(),
        playbar_dragging: false,
        cursor_ndc: None,
        pedal_brightness: 0.0,
        pedal_held: false,
        fps_accum: 0.0,
        fps_frame_count: 0,
        current_fps: 60.0,
        hovered_file: false,
        pending_file,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Application state (winit ApplicationHandler)
// ---------------------------------------------------------------------------

struct App {
    config: RenderConfig,
    event_rx: Receiver<NoteEvent>,
    clock: Arc<AudioClock>,
    transport_tx: Sender<TransportCommand>,
    transport_state: Arc<TransportState>,
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    visual_notes: Vec<VisualNote>,
    particles: ParticleSystem,
    last_frame: Instant,
    playbar_dragging: bool,
    cursor_ndc: Option<[f32; 2]>,
    pedal_brightness: f32,
    pedal_held: bool,
    // FPS counter state
    fps_accum: f32,
    fps_frame_count: u32,
    current_fps: f32,
    // File drag-and-drop
    hovered_file: bool,
    pending_file: Arc<Mutex<Option<PathBuf>>>,
}

/// All GPU resources: surface, device, pipelines, buffers, bloom textures.
struct GpuState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,

    // Pipelines
    note_pipeline: wgpu::RenderPipeline,
    key_pipeline: wgpu::RenderPipeline,
    particle_pipeline: wgpu::RenderPipeline,
    hitline_pipeline: wgpu::RenderPipeline,
    bloom_threshold_pipeline: wgpu::RenderPipeline,
    bloom_blur_h_pipeline: wgpu::RenderPipeline,
    bloom_blur_v_pipeline: wgpu::RenderPipeline,
    bloom_composite_pipeline: wgpu::RenderPipeline,

    // Buffers
    vertex_buffer: wgpu::Buffer,
    note_instance_buffer: wgpu::Buffer,
    key_instance_buffer: wgpu::Buffer,
    ui_instance_buffer: wgpu::Buffer,
    particle_instance_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,

    // Bind groups
    uniform_bind_group: wgpu::BindGroup,
    bloom_threshold_bind_group: wgpu::BindGroup,
    bloom_blur_h_bind_group: wgpu::BindGroup,
    bloom_blur_v_bind_group: wgpu::BindGroup,
    bloom_composite_bind_group: wgpu::BindGroup,

    // Offscreen textures for bloom
    scene_texture_view: wgpu::TextureView,
    bloom_a_texture_view: wgpu::TextureView,
    bloom_b_texture_view: wgpu::TextureView,

    // Pre-built piano key instances
    key_instances: Vec<KeyInstance>,

    _window: Arc<Window>,
}

// ---------------------------------------------------------------------------
// Piano key layout helper
// ---------------------------------------------------------------------------

pub fn is_black_key(midi: u8) -> bool {
    matches!(midi % 12, 1 | 3 | 6 | 8 | 10)
}

pub fn build_piano_keys() -> Vec<KeyInstance> {
    let mut keys = Vec::with_capacity(256);
    let white_width = 2.0 / 52.0; // 52 white keys in 88-key range (approx)
    let key_height = 0.15;
    let key_y = -0.92;

    // Shadow / depth base: a dark bar spanning the entire keyboard area
    keys.push(KeyInstance {
        position: [0.0, key_y - key_height * 0.48],
        size: [2.0, key_height * 0.12],
        color: [0.04, 0.04, 0.06, 0.9],
        border_radius: 0.0,
        _pad: [0.0; 3],
    });

    // Top-edge highlight
    keys.push(KeyInstance {
        position: [0.0, key_y + key_height * 0.52],
        size: [2.0, 0.003],
        color: [0.35, 0.38, 0.45, 0.3],
        border_radius: 0.0,
        _pad: [0.0; 3],
    });

    // First pass: white keys
    let mut white_idx = 0u32;
    for midi in PIANO_MIN..=PIANO_MAX {
        if is_black_key(midi) {
            continue;
        }
        let x = -1.0 + (white_idx as f32 + 0.5) * white_width;
        keys.push(KeyInstance {
            position: [x, key_y],
            size: [white_width * 0.92, key_height],
            color: [0.92, 0.92, 0.92, 1.0],
            border_radius: 0.18,
            _pad: [0.0; 3],
        });
        white_idx += 1;
    }

    // Second pass: black keys
    white_idx = 0;
    let mut prev_white_x = -1.0 + 0.5 * white_width;
    for midi in PIANO_MIN..=PIANO_MAX {
        if is_black_key(midi) {
            let x = prev_white_x + white_width * 0.5;
            keys.push(KeyInstance {
                position: [x, key_y + key_height * 0.18],
                size: [white_width * 0.55, key_height * 0.62],
                color: [0.12, 0.12, 0.16, 1.0],
                border_radius: 0.14,
                _pad: [0.0; 3],
            });
        } else {
            prev_white_x = -1.0 + (white_idx as f32 + 0.5) * white_width;
            white_idx += 1;
        }
    }

    // Third pass: key name indicators — small colored dots below each C note
    // and a dimmer dot under F notes for visual anchoring across the keyboard.
    white_idx = 0;
    for midi in PIANO_MIN..=PIANO_MAX {
        if is_black_key(midi) {
            continue;
        }
        let x = -1.0 + (white_idx as f32 + 0.5) * white_width;
        let pc = midi % 12;
        let dot_y = key_y - key_height * 0.55;
        let dot_size = 0.006;

        if pc == 0 {
            // C notes: bright cyan dot + register visualization
            let octave = midi / 12;
            let brightness = 0.5 + (octave as f32 - 1.0) * 0.08;
            keys.push(KeyInstance {
                position: [x, dot_y],
                size: [dot_size, dot_size],
                color: [0.2, brightness, 1.0, 0.9],
                border_radius: 0.5,
                _pad: [0.0; 3],
            });
        } else if pc == 5 {
            // F notes: dim reference dot
            keys.push(KeyInstance {
                position: [x, dot_y],
                size: [dot_size * 0.6, dot_size * 0.6],
                color: [0.3, 0.3, 0.35, 0.4],
                border_radius: 0.5,
                _pad: [0.0; 3],
            });
        }
        white_idx += 1;
    }

    keys
}

// ---------------------------------------------------------------------------
// Vertex buffer layouts
// ---------------------------------------------------------------------------

fn vertex_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: wgpu::VertexFormat::Float32x2,
        }],
    }
}

fn note_instance_layout() -> wgpu::VertexBufferLayout<'static> {
    static ATTRS: [wgpu::VertexAttribute; 6] = [
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32x2,
        }, // position
        wgpu::VertexAttribute {
            offset: 8,
            shader_location: 2,
            format: wgpu::VertexFormat::Float32x2,
        }, // size
        wgpu::VertexAttribute {
            offset: 16,
            shader_location: 3,
            format: wgpu::VertexFormat::Float32x4,
        }, // color
        wgpu::VertexAttribute {
            offset: 32,
            shader_location: 4,
            format: wgpu::VertexFormat::Float32,
        }, // border_radius
        wgpu::VertexAttribute {
            offset: 36,
            shader_location: 5,
            format: wgpu::VertexFormat::Float32,
        }, // glow_intensity
        wgpu::VertexAttribute {
            offset: 40,
            shader_location: 6,
            format: wgpu::VertexFormat::Float32x2,
        }, // _pad (skipped by shader, but stride needs it)
    ];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<NoteInstance>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &ATTRS,
    }
}

fn key_instance_layout() -> wgpu::VertexBufferLayout<'static> {
    static ATTRS: [wgpu::VertexAttribute; 5] = [
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32x2,
        },
        wgpu::VertexAttribute {
            offset: 8,
            shader_location: 2,
            format: wgpu::VertexFormat::Float32x2,
        },
        wgpu::VertexAttribute {
            offset: 16,
            shader_location: 3,
            format: wgpu::VertexFormat::Float32x4,
        },
        wgpu::VertexAttribute {
            offset: 32,
            shader_location: 4,
            format: wgpu::VertexFormat::Float32,
        },
        wgpu::VertexAttribute {
            offset: 36,
            shader_location: 5,
            format: wgpu::VertexFormat::Float32,
        }, // _pad
    ];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<KeyInstance>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &ATTRS,
    }
}

fn particle_instance_layout() -> wgpu::VertexBufferLayout<'static> {
    static ATTRS: [wgpu::VertexAttribute; 3] = [
        wgpu::VertexAttribute {
            offset: 0,
            shader_location: 1,
            format: wgpu::VertexFormat::Float32x2,
        },
        wgpu::VertexAttribute {
            offset: 8,
            shader_location: 2,
            format: wgpu::VertexFormat::Float32x2,
        },
        wgpu::VertexAttribute {
            offset: 16,
            shader_location: 3,
            format: wgpu::VertexFormat::Float32x4,
        },
    ];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<ParticleInstance>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &ATTRS,
    }
}

// ---------------------------------------------------------------------------
// GPU initialisation
// ---------------------------------------------------------------------------

impl GpuState {
    async fn new(window: Arc<Window>, config: &RenderConfig) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable GPU adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("shruti-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await
            .unwrap();

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // ---- compile shaders ----
        let geometry_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("notes.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/notes.wgsl").into()),
        });
        let fullscreen_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bloom.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bloom.wgsl").into()),
        });

        // ---- uniform buffer + bind group layout ----
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("frame-uniforms"),
            contents: bytemuck::bytes_of(&FrameUniforms {
                resolution: [size.width as f32, size.height as f32],
                time: 0.0,
                _pad: 0.0,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform-bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform-bg"),
            layout: &uniform_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let uniform_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("uniform-pl"),
                bind_group_layouts: &[&uniform_bgl],
                push_constant_ranges: &[],
            });

        // ---- offscreen textures for bloom ----
        let w = size.width.max(1);
        let h = size.height.max(1);
        let tex_usage =
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let tex_size = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };

        let scene_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("scene-tex"),
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: tex_usage,
            view_formats: &[],
        });
        let bloom_a_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bloom-a-tex"),
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: tex_usage,
            view_formats: &[],
        });
        let bloom_b_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bloom-b-tex"),
            size: tex_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: tex_usage,
            view_formats: &[],
        });

        let scene_texture_view = scene_texture.create_view(&Default::default());
        let bloom_a_texture_view = bloom_a_texture.create_view(&Default::default());
        let bloom_b_texture_view = bloom_b_texture.create_view(&Default::default());

        let bloom_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bloom-sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // ---- bloom bind group layouts ----
        // For threshold/blur passes: texture + sampler
        let bloom_tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bloom-tex-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // For composite: bloom_tex + sampler + scene_tex
        let bloom_composite_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bloom-composite-bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        // Bind groups for bloom passes
        let bloom_threshold_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom-threshold-bg"),
            layout: &bloom_tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&scene_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&bloom_sampler),
                },
            ],
        });

        let bloom_blur_h_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom-blur-h-bg"),
            layout: &bloom_tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bloom_a_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&bloom_sampler),
                },
            ],
        });

        let bloom_blur_v_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom-blur-v-bg"),
            layout: &bloom_tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bloom_b_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&bloom_sampler),
                },
            ],
        });

        let bloom_composite_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom-composite-bg"),
            layout: &bloom_composite_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&bloom_a_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&bloom_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&scene_texture_view),
                },
            ],
        });

        // ---- pipeline layouts ----
        let bloom_tex_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom-tex-pl"),
            bind_group_layouts: &[&bloom_tex_bgl, &uniform_bgl],
            push_constant_ranges: &[],
        });
        let bloom_composite_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bloom-composite-pl"),
            bind_group_layouts: &[&bloom_composite_bgl, &uniform_bgl],
            push_constant_ranges: &[],
        });

        // ---- helper for fullscreen pipelines ----
        let make_fullscreen_pipeline =
            |label: &str,
             layout: &wgpu::PipelineLayout,
             fs_entry: &str,
             blend: Option<wgpu::BlendState>| {
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(layout),
                    vertex: wgpu::VertexState {
                        module: &fullscreen_shader,
                        entry_point: Some("vs_fullscreen"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &fullscreen_shader,
                        entry_point: Some(fs_entry),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                })
            };

        // ---- instanced geometry pipelines ----
        let note_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("note-pipeline"),
            layout: Some(&uniform_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &geometry_shader,
                entry_point: Some("vs_notes"),
                buffers: &[vertex_layout(), note_instance_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &geometry_shader,
                entry_point: Some("fs_notes"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let key_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("key-pipeline"),
            layout: Some(&uniform_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &geometry_shader,
                entry_point: Some("vs_keys"),
                buffers: &[vertex_layout(), key_instance_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &geometry_shader,
                entry_point: Some("fs_keys"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let particle_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle-pipeline"),
            layout: Some(&uniform_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &geometry_shader,
                entry_point: Some("vs_particles"),
                buffers: &[vertex_layout(), particle_instance_layout()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &geometry_shader,
                entry_point: Some("fs_particles"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::One, // additive
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Fullscreen hit-line overlay (drawn on top of scene; alpha blend)
        let hitline_pipeline = make_fullscreen_pipeline(
            "hitline-pipeline",
            &bloom_tex_pl, // layout doesn't matter, hitline shader doesn't sample textures — but we need *a* layout
            "fs_hitline",
            Some(wgpu::BlendState::ALPHA_BLENDING),
        );

        // Bloom passes
        let bloom_threshold_pipeline =
            make_fullscreen_pipeline("bloom-threshold", &bloom_tex_pl, "fs_bloom_threshold", None);
        let bloom_blur_h_pipeline =
            make_fullscreen_pipeline("bloom-blur-h", &bloom_tex_pl, "fs_bloom_blur_h", None);
        let bloom_blur_v_pipeline =
            make_fullscreen_pipeline("bloom-blur-v", &bloom_tex_pl, "fs_bloom_blur_v", None);
        let bloom_composite_pipeline = make_fullscreen_pipeline(
            "bloom-composite",
            &bloom_composite_pl,
            "fs_bloom_composite",
            None,
        );

        // ---- vertex / instance buffers ----
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad-verts"),
            contents: bytemuck::cast_slice(&QUAD_VERTS),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let note_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("note-instances"),
            size: (MAX_NOTE_INSTANCES * std::mem::size_of::<NoteInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let key_instances = build_piano_keys();
        let key_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("key-instances"),
            size: (MAX_KEY_INSTANCES * std::mem::size_of::<KeyInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ui_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui-instances"),
            size: (MAX_UI_INSTANCES * std::mem::size_of::<KeyInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let particle_instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particle-instances"),
            size: (MAX_PARTICLE_INSTANCES * std::mem::size_of::<ParticleInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let _ = config; // consumed above via bloom_enabled / particles_enabled checks

        tracing::info!("GPU initialisation complete — all pipelines created");

        Self {
            surface,
            device,
            queue,
            surface_config,
            note_pipeline,
            key_pipeline,
            particle_pipeline,
            hitline_pipeline,
            bloom_threshold_pipeline,
            bloom_blur_h_pipeline,
            bloom_blur_v_pipeline,
            bloom_composite_pipeline,
            vertex_buffer,
            note_instance_buffer,
            key_instance_buffer,
            ui_instance_buffer,
            particle_instance_buffer,
            uniform_buffer,
            uniform_bind_group,
            bloom_threshold_bind_group,
            bloom_blur_h_bind_group,
            bloom_blur_v_bind_group,
            bloom_composite_bind_group,
            scene_texture_view,
            bloom_a_texture_view,
            bloom_b_texture_view,
            key_instances,
            _window: window,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.surface_config.width = new_size.width;
            self.surface_config.height = new_size.height;
            self.surface.configure(&self.device, &self.surface_config);
            // NOTE: bloom textures should be recreated here for correct sizing.
            // Omitted for brevity in this iteration — bloom will render at init
            // resolution. Full resize rebuild is a follow-up.
        }
    }
}

// ---------------------------------------------------------------------------
// winit event loop
// ---------------------------------------------------------------------------

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title("Shruti Parade — Piano Transcription Visualiser")
            .with_inner_size(LogicalSize::new(self.config.width, self.config.height));

        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let gpu = pollster::block_on(GpuState::new(window.clone(), &self.config));

        self.window = Some(window);
        self.gpu = Some(gpu);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(size);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_ndc = self.cursor_to_ndc(position);
                if self.playbar_dragging {
                    if let Some(ndc) = self.cursor_ndc {
                        self.seek_from_ndc(ndc);
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    if state == ElementState::Pressed {
                        if let Some(ndc) = self.cursor_ndc {
                            self.handle_playbar_press(ndc);
                        }
                    } else {
                        self.playbar_dragging = false;
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed {
                    self.handle_key_input(event.logical_key);
                }
            }
            WindowEvent::DroppedFile(path) => {
                self.hovered_file = false;
                tracing::info!("File dropped: {}", path.display());
                *self.pending_file.lock().unwrap() = Some(path);
                event_loop.exit();
            }
            WindowEvent::HoveredFile(_) => {
                self.hovered_file = true;
            }
            WindowEvent::HoveredFileCancelled => {
                self.hovered_file = false;
            }
            WindowEvent::RedrawRequested => {
                // Check if rfd picker thread delivered a file
                if self.pending_file.lock().unwrap().is_some() {
                    event_loop.exit();
                    return;
                }
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32();
                self.last_frame = now;

                // FPS tracking (update once per second)
                self.fps_accum += dt;
                self.fps_frame_count += 1;
                if self.fps_accum >= 1.0 {
                    self.current_fps = self.fps_frame_count as f32 / self.fps_accum;
                    self.fps_accum = 0.0;
                    self.fps_frame_count = 0;
                }

                self.update_notes(dt);
                self.render_frame();

                if let Some(w) = self.window.as_ref() {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Note tracking + particle spawning
// ---------------------------------------------------------------------------

impl App {
    fn playbar_layout(&self) -> PlaybarLayout {
        let bar = Rect {
            center: [0.0, PLAYBAR_Y],
            size: [PLAYBAR_WIDTH, PLAYBAR_HEIGHT],
        };
        let button = Rect {
            center: [PLAY_BUTTON_X, PLAYBAR_Y],
            size: [PLAY_BUTTON_SIZE, PLAY_BUTTON_SIZE],
        };
        PlaybarLayout { bar, button }
    }

    fn cursor_to_ndc(&self, position: winit::dpi::PhysicalPosition<f64>) -> Option<[f32; 2]> {
        let window = self.window.as_ref()?;
        let size = window.inner_size();
        if size.width == 0 || size.height == 0 {
            return None;
        }
        let x = (position.x as f32 / size.width as f32) * 2.0 - 1.0;
        let y = 1.0 - (position.y as f32 / size.height as f32) * 2.0;
        Some([x, y])
    }

    fn clear_visuals(&mut self) {
        self.visual_notes.clear();
        self.particles.clear();
        self.pedal_brightness = 0.0;
        self.pedal_held = false;
        // Drain stale events that were queued before the seek.
        while self.event_rx.try_recv().is_ok() {}
    }

    fn handle_playbar_press(&mut self, ndc: [f32; 2]) {
        let layout = self.playbar_layout();
        // Play/pause button always works (even before total_samples is known)
        if layout.button.contains(ndc) {
            self.toggle_pause();
            return;
        }
        // File open button
        let file_btn = Rect {
            center: [FILE_BUTTON_X, FILE_BUTTON_Y],
            size: [FILE_BUTTON_SIZE, FILE_BUTTON_SIZE],
        };
        if file_btn.contains(ndc) {
            self.open_file_picker();
            return;
        }
        if !self.can_seek() {
            return;
        }
        if layout.bar.contains(ndc) {
            self.playbar_dragging = true;
            self.seek_from_ndc(ndc);
        }
    }

    fn seek_from_ndc(&mut self, ndc: [f32; 2]) {
        let total_samples = self.transport_state.total_samples();
        if total_samples == 0 {
            return;
        }
        let layout = self.playbar_layout();
        let bar_left = layout.bar.left() + PLAYBAR_PADDING;
        let bar_right = layout.bar.left() + layout.bar.size[0] - PLAYBAR_PADDING;
        let t = ((ndc[0] - bar_left) / (bar_right - bar_left)).clamp(0.0, 1.0);
        let target = (t as f64 * total_samples as f64).round() as u64;
        self.seek_to_samples(target);
    }

    fn seek_to_samples(&mut self, sample: u64) {
        if !self.can_seek() {
            return;
        }
        let total_samples = self.transport_state.total_samples();
        let target = sample.min(total_samples);
        // Flush stale audio from the ring immediately so the output callback
        // doesn't play old audio between now and when the feed thread resets.
        self.transport_state.request_flush();
        self.clock.set_samples(target);
        self.clear_visuals();
        let _ = self
            .transport_tx
            .try_send(TransportCommand::SeekSamples(target));
    }

    fn seek_by_seconds(&mut self, delta_seconds: f64) {
        if !self.can_seek() {
            return;
        }
        let sr = self.clock.sample_rate();
        let delta_samples = (delta_seconds * sr) as i64;
        let current = self.clock.now_samples() as i64;
        let total = self.transport_state.total_samples() as i64;
        let target = (current + delta_samples).clamp(0, total) as u64;
        self.seek_to_samples(target);
    }

    fn set_paused(&mut self, paused: bool) {
        self.transport_state.set_paused(paused);
        let cmd = if paused {
            TransportCommand::Pause
        } else {
            TransportCommand::Play
        };
        let _ = self.transport_tx.try_send(cmd);
    }

    fn toggle_pause(&mut self) {
        let paused = !self.transport_state.is_paused();
        self.set_paused(paused);
    }

    fn open_file_picker(&self) {
        let pending = self.pending_file.clone();
        std::thread::Builder::new()
            .name("file-picker".into())
            .spawn(move || {
                let file = rfd::FileDialog::new()
                    .add_filter(
                        "Audio / MIDI",
                        &["mid", "midi", "wav", "mp3", "flac", "ogg"],
                    )
                    .pick_file();
                if let Some(path) = file {
                    tracing::info!("File picked: {}", path.display());
                    *pending.lock().unwrap() = Some(path);
                }
            })
            .ok();
    }

    fn handle_key_input(&mut self, key: Key) {
        match key {
            Key::Named(NamedKey::Space) => self.toggle_pause(),
            _ if !self.can_seek() => {}
            Key::Named(NamedKey::ArrowLeft) => self.seek_by_seconds(-PLAYBAR_SEEK_SECONDS),
            Key::Named(NamedKey::ArrowRight) => self.seek_by_seconds(PLAYBAR_SEEK_SECONDS),
            Key::Named(NamedKey::Home) => self.seek_to_samples(0),
            Key::Named(NamedKey::End) => {
                let total = self.transport_state.total_samples();
                self.seek_to_samples(total);
            }
            _ => {}
        }
    }

    fn can_seek(&self) -> bool {
        self.transport_state.total_samples() > 0
    }

    fn update_notes(&mut self, dt: f32) {
        while let Ok(ev) = self.event_rx.try_recv() {
            let t = self.clock.sample_to_seconds(ev.sample_time);
            match ev.kind {
                NoteEventKind::NoteOn => {
                    self.visual_notes.push(VisualNote {
                        pitch: ev.pitch,
                        velocity: ev.velocity,
                        start_time: t,
                        end_time: None,
                    });

                    // Spawn particles at the hit position
                    if self.config.particles_enabled {
                        let x = pitch_to_ndc_x(ev.pitch);
                        let hue = pitch_to_hue(ev.pitch);
                        let (r, g, b) = hsv_to_rgb(hue, 0.9, 1.0);
                        let a = (ev.velocity as f32 / 127.0).clamp(0.4, 1.0);
                        self.particles
                            .spawn([x, HIT_LINE_Y], [r, g, b, a], PARTICLE_BURST);
                    }
                }
                NoteEventKind::NoteOff => {
                    if let Some(n) = self
                        .visual_notes
                        .iter_mut()
                        .rev()
                        .find(|n| n.pitch == ev.pitch && n.end_time.is_none())
                    {
                        n.end_time = Some(t);
                    }
                }
                NoteEventKind::PedalOn => {
                    self.pedal_brightness = 1.0;
                    self.pedal_held = true;
                }
                NoteEventKind::PedalOff => {
                    self.pedal_held = false;
                }
            }
        }

        // Fade pedal brightness when released (same pattern as particle fade)
        if self.pedal_held {
            self.pedal_brightness = 1.0;
        } else if self.pedal_brightness > 0.0 {
            self.pedal_brightness = (self.pedal_brightness - dt * 2.5).max(0.0);
        }

        // Advance particles
        self.particles.tick(dt);

        // Cull old notes
        let now = self.clock.now_seconds();
        self.visual_notes
            .retain(|n| now - n.start_time < NOTE_CULL_SECONDS);
    }

    fn render_frame(&self) {
        let gpu = match self.gpu.as_ref() {
            Some(g) => g,
            None => return,
        };

        let now = self.clock.now_seconds();
        let w = gpu.surface_config.width as f32;
        let h = gpu.surface_config.height as f32;

        // ---- update uniforms ----
        gpu.queue.write_buffer(
            &gpu.uniform_buffer,
            0,
            bytemuck::bytes_of(&FrameUniforms {
                resolution: [w, h],
                time: now as f32,
                _pad: 0.0,
            }),
        );

        // ---- collect which pitches are currently active ----
        let mut active_pitches = [false; 128];
        for note in &self.visual_notes {
            if note.end_time.is_none() && note.pitch >= PIANO_MIN && note.pitch <= PIANO_MAX {
                active_pitches[note.pitch as usize] = true;
            }
        }

        // ---- update key instances with highlighting ----
        let key_draw_count;
        {
            let mut keys = gpu.key_instances.clone();
            // First 2 entries are decorative (shadow + highlight), then 52 white, then 36 black
            let deco_offset = 2;
            let mut white_idx: usize = 0;
            let mut black_idx: usize = 0;
            for midi in PIANO_MIN..=PIANO_MAX {
                if is_black_key(midi) {
                    let ki = deco_offset + 52 + black_idx;
                    if ki < keys.len() && active_pitches[midi as usize] {
                        let hue = pitch_to_hue(midi);
                        let (r, g, b) = hsv_to_rgb(hue, 0.9, 0.9);
                        keys[ki].color = [r, g, b, 1.0];
                    } else if ki < keys.len() {
                        keys[ki].color = [0.12, 0.12, 0.16, 1.0];
                    }
                    black_idx += 1;
                } else {
                    let ki = deco_offset + white_idx;
                    if ki < keys.len() && active_pitches[midi as usize] {
                        let hue = pitch_to_hue(midi);
                        let (r, g, b) = hsv_to_rgb(hue, 0.7, 1.0);
                        keys[ki].color = [r, g, b, 1.0];
                    } else if ki < keys.len() {
                        keys[ki].color = [0.92, 0.92, 0.92, 1.0];
                    }
                    white_idx += 1;
                }
            }

            // Pedal indicator bar — thin bar below the piano keys.
            // Rendered through the same key pipeline (with bloom).
            // Fades out smoothly when pedal is released.
            if self.pedal_brightness > 0.001 {
                let b = self.pedal_brightness;
                keys.push(KeyInstance {
                    position: [0.0, -0.995],
                    size: [1.90, 0.015],
                    color: [0.35 * b, 0.70 * b, 1.0 * b, b.clamp(0.0, 1.0)],
                    border_radius: 0.3,
                    _pad: [0.0; 3],
                });
            }

            // ---- Pitch confidence heat bar ----
            // Thin colored bars above the piano keyboard showing per-pitch
            // activity. Active notes glow brightly; recently-released notes fade.
            let heat_y = -0.835; // just above piano keys
            let heat_h = 0.012;
            let key_w = 2.0 / 52.0;
            for note in &self.visual_notes {
                if note.pitch < PIANO_MIN || note.pitch > PIANO_MAX {
                    continue;
                }
                let intensity = if let Some(end) = note.end_time {
                    // Fade out recently-released notes
                    let elapsed = (now - end) as f32;
                    (1.0 - elapsed * 2.0).max(0.0) * 0.5
                } else {
                    (note.velocity as f32 / 127.0).clamp(0.3, 1.0)
                };
                if intensity < 0.01 {
                    continue;
                }
                let x = pitch_to_ndc_x(note.pitch);
                let hue = pitch_to_hue(note.pitch);
                let (r, g, b) = hsv_to_rgb(hue, 0.8, intensity);
                keys.push(KeyInstance {
                    position: [x, heat_y],
                    size: [key_w * 0.7, heat_h],
                    color: [r, g, b, intensity],
                    border_radius: 0.2,
                    _pad: [0.0; 3],
                });
            }

            key_draw_count = keys.len().min(MAX_KEY_INSTANCES);
            gpu.queue
                .write_buffer(&gpu.key_instance_buffer, 0, bytemuck::cast_slice(&keys));
        }

        // ---- build note instances ----
        let mut note_instances = Vec::with_capacity(self.visual_notes.len());
        let mut active_note_count: u32 = 0;
        for note in &self.visual_notes {
            if note.pitch < PIANO_MIN || note.pitch > PIANO_MAX {
                continue;
            }

            let x = pitch_to_ndc_x(note.pitch);
            let white_w = 2.0 / 52.0;
            let note_w = if is_black_key(note.pitch) {
                white_w * 0.55 * 0.85
            } else {
                white_w * 0.85
            };

            let y_top = HIT_LINE_Y + (now - note.start_time) as f32 * SCROLL_SPEED;
            let y_bot = match note.end_time {
                Some(end) => HIT_LINE_Y + (now - end) as f32 * SCROLL_SPEED,
                None => HIT_LINE_Y,
            };

            // Minimum note height for visibility (short staccato notes)
            let note_h = (y_top - y_bot).max(0.02);
            let y_c = (y_top + y_bot) / 2.0;

            let hue = pitch_to_hue(note.pitch);
            // Velocity → brightness: louder notes are brighter
            let vel_norm = (note.velocity as f32 / 127.0).clamp(0.3, 1.0);
            let brightness = 0.55 + 0.45 * vel_norm;
            let (r, g, b) = hsv_to_rgb(hue, 0.80, brightness);
            let a = 0.5 + 0.5 * vel_norm;

            let is_active = note.end_time.is_none();
            if is_active {
                active_note_count += 1;
            }

            note_instances.push(NoteInstance {
                position: [x, y_c],
                size: [note_w, note_h],
                color: [r, g, b, a],
                border_radius: 0.3,
                glow_intensity: if is_active { 1.2 } else { 0.12 },
                _pad: [0.0; 2],
            });

            // ---- Note label accent strip ----
            // Bright thin bar at bottom of each note, using a 12-tone
            // pitch-class colour palette for quick visual identification.
            let label_h = 0.012_f32.min(note_h * 0.3);
            let label_y = y_bot + label_h * 0.5;
            let pc = note.pitch % 12;
            let label_hue = pc as f32 * 30.0; // 30° steps for 12 classes
            let (lr, lg, lb) = hsv_to_rgb(label_hue, 1.0, 1.0);
            note_instances.push(NoteInstance {
                position: [x, label_y],
                size: [note_w, label_h],
                color: [lr, lg, lb, a.min(0.9)],
                border_radius: 0.0,
                glow_intensity: 0.0,
                _pad: [0.0; 2],
            });
        }
        let note_count = note_instances.len().min(MAX_NOTE_INSTANCES);
        if note_count > 0 {
            gpu.queue.write_buffer(
                &gpu.note_instance_buffer,
                0,
                bytemuck::cast_slice(&note_instances[..note_count]),
            );
        }

        // ---- build particle instances ----
        let particle_instances = self.particles.instances();
        let particle_count = particle_instances.len().min(MAX_PARTICLE_INSTANCES);
        if particle_count > 0 {
            gpu.queue.write_buffer(
                &gpu.particle_instance_buffer,
                0,
                bytemuck::cast_slice(&particle_instances[..particle_count]),
            );
        }

        // ---- build playbar instances ----
        let mut ui_instances: Vec<KeyInstance> = Vec::new();
        let total_samples = self.transport_state.total_samples();
        let can_seek = total_samples > 0;
        let is_paused = self.transport_state.is_paused();
        let layout = self.playbar_layout();

        // ── Top bar background (dark translucent strip) ──
        ui_instances.push(KeyInstance {
            position: [0.0, 0.93],
            size: [2.0, 0.16],
            color: [0.03, 0.03, 0.06, 0.70],
            border_radius: 0.0,
            _pad: [0.0; 3],
        });

        // ── Progress bar background ──
        let bar_color = if can_seek {
            [0.08, 0.09, 0.13, 0.85]
        } else {
            [0.06, 0.06, 0.08, 0.5]
        };
        ui_instances.push(KeyInstance {
            position: layout.bar.center,
            size: layout.bar.size,
            color: bar_color,
            border_radius: 0.35,
            _pad: [0.0; 3],
        });

        if can_seek {
            let progress = (self.clock.now_samples().min(total_samples) as f32)
                / (total_samples as f32).max(1.0);
            let bar_left = layout.bar.left() + PLAYBAR_PADDING;
            let bar_width = (layout.bar.size[0] - PLAYBAR_PADDING * 2.0).max(0.0);

            // Gradient progress fill — 4 segments that shift hue across the bar
            let fill_w_total = (bar_width * progress).max(0.002);
            let n_segments: usize = 4;
            let seg_w = fill_w_total / n_segments as f32;
            for seg in 0..n_segments {
                let t = seg as f32 / n_segments as f32;
                let seg_left = bar_left + seg as f32 * seg_w;
                let seg_right = seg_left + seg_w;
                if seg_right <= bar_left {
                    continue;
                }
                let seg_cx = (seg_left + seg_right) * 0.5;
                // Gradient from cyan to blue-violet across the fill
                let hue = 190.0 + t * 70.0;
                let (gr, gg, gb) = hsv_to_rgb(hue, 0.8, 0.85);
                ui_instances.push(KeyInstance {
                    position: [seg_cx, layout.bar.center[1]],
                    size: [seg_w, layout.bar.size[1] * 0.55],
                    color: [gr, gg, gb, 0.9],
                    border_radius: if seg == 0 || seg == n_segments - 1 {
                        0.35
                    } else {
                        0.0
                    },
                    _pad: [0.0; 3],
                });
            }

            // Playback handle (scrubber)
            let handle_x = bar_left + bar_width * progress;
            ui_instances.push(KeyInstance {
                position: [handle_x, layout.bar.center[1]],
                size: [PLAYBAR_HANDLE_WIDTH, layout.bar.size[1] * 0.9],
                color: [0.9, 0.9, 0.95, 0.95],
                border_radius: 0.3,
                _pad: [0.0; 3],
            });
        }

        // ── State-coloured play/pause button ──
        let btn_color = if !can_seek {
            [0.12, 0.12, 0.16, 0.5]
        } else if is_paused {
            [0.15, 0.35, 0.18, 0.9] // green tint (ready to play)
        } else {
            [0.40, 0.28, 0.10, 0.9] // amber tint (playing)
        };
        ui_instances.push(KeyInstance {
            position: layout.button.center,
            size: layout.button.size,
            color: btn_color,
            border_radius: 0.35,
            _pad: [0.0; 3],
        });

        // Always show play/pause icon, even before total_samples is known
        if is_paused {
            // Play triangle — rendered as a right-pointing arrow using
            // a wide rectangle (approximation via rounded rect)
            let tri_w = layout.button.size[0] * 0.30;
            let tri_h = layout.button.size[1] * 0.50;
            ui_instances.push(KeyInstance {
                position: [
                    layout.button.center[0] + tri_w * 0.08,
                    layout.button.center[1],
                ],
                size: [tri_w, tri_h],
                color: [0.85, 0.92, 1.0, 0.95],
                border_radius: 0.25,
                _pad: [0.0; 3],
            });
        } else {
            // Pause bars
            let offset = layout.button.size[0] * 0.16;
            let bar_w = layout.button.size[0] * 0.14;
            let bar_h = layout.button.size[1] * 0.50;
            ui_instances.push(KeyInstance {
                position: [layout.button.center[0] - offset, layout.button.center[1]],
                size: [bar_w, bar_h],
                color: [0.85, 0.92, 1.0, 0.95],
                border_radius: 0.15,
                _pad: [0.0; 3],
            });
            ui_instances.push(KeyInstance {
                position: [layout.button.center[0] + offset, layout.button.center[1]],
                size: [bar_w, bar_h],
                color: [0.85, 0.92, 1.0, 0.95],
                border_radius: 0.15,
                _pad: [0.0; 3],
            });
        }

        // ---- FPS counter (top-right) ----
        {
            let fps_fraction = (self.current_fps / 120.0).clamp(0.0, 1.0);
            let bar_max_w = 0.12;
            let bar_w = bar_max_w * fps_fraction;
            let bar_h = 0.015;
            let bar_x = 0.93 - (bar_max_w - bar_w) * 0.5;
            let bar_y = 0.97;
            let t = (self.current_fps / 60.0).clamp(0.0, 1.0);
            let (fr, fg, fb) = if t < 0.5 {
                let s = t * 2.0;
                (0.9, 0.2 + 0.6 * s, 0.2)
            } else {
                let s = (t - 0.5) * 2.0;
                (0.9 - 0.7 * s, 0.8 + 0.1 * s, 0.2 + 0.1 * s)
            };
            // Background
            ui_instances.push(KeyInstance {
                position: [0.93, bar_y],
                size: [bar_max_w, bar_h],
                color: [0.1, 0.1, 0.12, 0.6],
                border_radius: 0.2,
                _pad: [0.0; 3],
            });
            // Fill
            ui_instances.push(KeyInstance {
                position: [bar_x, bar_y],
                size: [bar_w, bar_h * 0.7],
                color: [fr, fg, fb, 0.85],
                border_radius: 0.2,
                _pad: [0.0; 3],
            });
            // FPS digit indicator: series of tiny dots representing tens digit
            let fps_tens = (self.current_fps / 10.0).round() as u32;
            let dot_count = fps_tens.min(12);
            let dot_x_start = 0.93 - bar_max_w * 0.5;
            let dot_gap = bar_max_w / 13.0;
            for d in 0..dot_count {
                ui_instances.push(KeyInstance {
                    position: [dot_x_start + d as f32 * dot_gap, bar_y - 0.015],
                    size: [0.004, 0.004],
                    color: [fr, fg, fb, 0.7],
                    border_radius: 0.5,
                    _pad: [0.0; 3],
                });
            }
        }

        // ---- BPM estimate (top-right, below FPS) ----
        // Estimate BPM from recent note onset intervals
        {
            let mut recent_onsets: Vec<f64> = self
                .visual_notes
                .iter()
                .filter(|n| now - n.start_time < 8.0)
                .map(|n| n.start_time)
                .collect();
            recent_onsets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            recent_onsets.dedup();

            if recent_onsets.len() >= 3 {
                let mut intervals: Vec<f64> = recent_onsets
                    .windows(2)
                    .map(|w| w[1] - w[0])
                    .filter(|&d| d > 0.1 && d < 2.0)
                    .collect();
                if !intervals.is_empty() {
                    intervals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let median = intervals[intervals.len() / 2];
                    let bpm = (60.0 / median).round() as u32;
                    let bpm_clamped = bpm.clamp(30, 300);
                    // Show BPM as a horizontal bar sized proportionally
                    let bpm_frac = (bpm_clamped as f32 - 30.0) / 270.0;
                    let bpm_bar_w = 0.08 * bpm_frac;
                    // Background pill
                    ui_instances.push(KeyInstance {
                        position: [0.93, 0.935],
                        size: [0.10, 0.015],
                        color: [0.06, 0.06, 0.08, 0.50],
                        border_radius: 0.3,
                        _pad: [0.0; 3],
                    });
                    // BPM fill
                    ui_instances.push(KeyInstance {
                        position: [0.93 - (0.08 - bpm_bar_w) * 0.5, 0.935],
                        size: [bpm_bar_w.max(0.005), 0.010],
                        color: [0.9, 0.55, 0.2, 0.8],
                        border_radius: 0.3,
                        _pad: [0.0; 3],
                    });
                }
            }
        }

        // ---- File open button ----
        ui_instances.push(KeyInstance {
            position: [FILE_BUTTON_X, FILE_BUTTON_Y],
            size: [FILE_BUTTON_SIZE, FILE_BUTTON_SIZE],
            color: [0.15, 0.16, 0.20, 0.9],
            border_radius: 0.35,
            _pad: [0.0; 3],
        });
        // "F" letter — approximated as a vertical bar + two horizontal bars
        {
            let cx = FILE_BUTTON_X;
            let cy = FILE_BUTTON_Y;
            let s = FILE_BUTTON_SIZE;
            // Vertical bar (left side of F)
            ui_instances.push(KeyInstance {
                position: [cx - s * 0.10, cy],
                size: [s * 0.10, s * 0.50],
                color: [0.85, 0.92, 1.0, 0.95],
                border_radius: 0.1,
                _pad: [0.0; 3],
            });
            // Top horizontal bar
            ui_instances.push(KeyInstance {
                position: [cx + s * 0.05, cy + s * 0.20],
                size: [s * 0.30, s * 0.08],
                color: [0.85, 0.92, 1.0, 0.95],
                border_radius: 0.1,
                _pad: [0.0; 3],
            });
            // Middle horizontal bar
            ui_instances.push(KeyInstance {
                position: [cx + s * 0.01, cy],
                size: [s * 0.22, s * 0.08],
                color: [0.85, 0.92, 1.0, 0.95],
                border_radius: 0.1,
                _pad: [0.0; 3],
            });
        }

        // ---- Active note count indicator (top-left) ----
        {
            let max_dots: u32 = 10;
            let dot_w = 0.012;
            let dot_h = 0.012;
            let dot_y = 0.97;
            let dot_start_x = -0.93_f32;
            let dot_gap = 0.016;
            let active = active_note_count.min(max_dots);

            // Background track
            ui_instances.push(KeyInstance {
                position: [dot_start_x + (max_dots as f32 * dot_gap) * 0.5, dot_y],
                size: [max_dots as f32 * dot_gap + dot_w, dot_h * 1.4],
                color: [0.06, 0.06, 0.08, 0.45],
                border_radius: 0.2,
                _pad: [0.0; 3],
            });

            for i in 0..max_dots {
                let x = dot_start_x + i as f32 * dot_gap;
                let lit = i < active;
                let intensity = if lit { 0.9 } else { 0.15 };
                let (sr, sg, sb) = if lit {
                    hsv_to_rgb(200.0 + i as f32 * 12.0, 0.7, intensity)
                } else {
                    (0.15, 0.15, 0.18)
                };
                ui_instances.push(KeyInstance {
                    position: [x, dot_y],
                    size: [dot_w * 0.7, dot_h * 0.7],
                    color: [sr, sg, sb, if lit { 0.9 } else { 0.3 }],
                    border_radius: 0.5,
                    _pad: [0.0; 3],
                });
            }
        }

        // ---- Level meter (top-left, below polyphony dots) ----
        // Approximates audio level from the max velocity of active notes.
        {
            let max_vel: f32 = self
                .visual_notes
                .iter()
                .filter(|n| n.end_time.is_none())
                .map(|n| n.velocity as f32 / 127.0)
                .fold(0.0f32, f32::max);

            let meter_x = -0.93;
            let meter_y = 0.935;
            let meter_w = 0.15;
            let meter_h = 0.010;
            let n_bars: usize = 8;
            let bar_gap = meter_w / n_bars as f32;

            // Background track
            ui_instances.push(KeyInstance {
                position: [meter_x + meter_w * 0.5, meter_y],
                size: [meter_w + 0.01, meter_h * 1.4],
                color: [0.05, 0.05, 0.07, 0.40],
                border_radius: 0.2,
                _pad: [0.0; 3],
            });

            for i in 0..n_bars {
                let t = (i + 1) as f32 / n_bars as f32;
                let lit = max_vel >= t - 0.5 / n_bars as f32;
                let x = meter_x + i as f32 * bar_gap;
                // Green → yellow → red gradient
                let (mr, mg, mb) = if t < 0.5 {
                    (0.2, 0.8, 0.3)
                } else if t < 0.8 {
                    (0.8, 0.7, 0.1)
                } else {
                    (0.9, 0.2, 0.15)
                };
                let alpha = if lit { 0.85 } else { 0.15 };
                ui_instances.push(KeyInstance {
                    position: [x, meter_y],
                    size: [bar_gap * 0.7, meter_h * 0.7],
                    color: [mr, mg, mb, alpha],
                    border_radius: 0.15,
                    _pad: [0.0; 3],
                });
            }
        }

        // ---- Track info bar (top centre) ----
        if can_seek {
            let bar_w = 0.40;
            let bar_h = 0.025;
            let bar_y = 0.97;
            // Background pill
            ui_instances.push(KeyInstance {
                position: [0.0, bar_y],
                size: [bar_w, bar_h],
                color: [0.05, 0.06, 0.09, 0.65],
                border_radius: 0.4,
                _pad: [0.0; 3],
            });

            // Elapsed / total represented as a gradient fill
            let progress = (self.clock.now_samples().min(total_samples) as f32)
                / (total_samples as f32).max(1.0);
            let inner_w = bar_w - 0.02;
            let fill_w = inner_w * progress;
            let fill_x = -(inner_w) * 0.5 + fill_w * 0.5;
            // Fill gradient: dim at start, brighter at current position
            ui_instances.push(KeyInstance {
                position: [fill_x, bar_y],
                size: [fill_w.max(0.001), bar_h * 0.30],
                color: [0.25, 0.50, 0.80, 0.50],
                border_radius: 0.4,
                _pad: [0.0; 3],
            });

            // Small bright dot at current position
            let dot_x = -(inner_w) * 0.5 + inner_w * progress;
            ui_instances.push(KeyInstance {
                position: [dot_x, bar_y],
                size: [0.006, bar_h * 0.5],
                color: [0.7, 0.85, 1.0, 0.9],
                border_radius: 0.5,
                _pad: [0.0; 3],
            });
        }

        // ---- Drag-and-drop cyan border ----
        if self.hovered_file {
            let bw = 0.01; // border width in NDC
                           // Top
            ui_instances.push(KeyInstance {
                position: [0.0, 1.0 - bw * 0.5],
                size: [2.0, bw],
                color: [0.0, 0.85, 1.0, 0.9],
                border_radius: 0.0,
                _pad: [0.0; 3],
            });
            // Bottom
            ui_instances.push(KeyInstance {
                position: [0.0, -1.0 + bw * 0.5],
                size: [2.0, bw],
                color: [0.0, 0.85, 1.0, 0.9],
                border_radius: 0.0,
                _pad: [0.0; 3],
            });
            // Left
            ui_instances.push(KeyInstance {
                position: [-1.0 + bw * 0.5, 0.0],
                size: [bw, 2.0],
                color: [0.0, 0.85, 1.0, 0.9],
                border_radius: 0.0,
                _pad: [0.0; 3],
            });
            // Right
            ui_instances.push(KeyInstance {
                position: [1.0 - bw * 0.5, 0.0],
                size: [bw, 2.0],
                color: [0.0, 0.85, 1.0, 0.9],
                border_radius: 0.0,
                _pad: [0.0; 3],
            });
        }

        let ui_count = ui_instances.len().min(MAX_UI_INSTANCES);
        if ui_count > 0 {
            gpu.queue.write_buffer(
                &gpu.ui_instance_buffer,
                0,
                bytemuck::cast_slice(&ui_instances[..ui_count]),
            );
        }

        // ---- acquire surface texture ----
        let output = match gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!("Surface error: {e:?}");
                gpu.surface.configure(&gpu.device, &gpu.surface_config);
                return;
            }
        };
        let final_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame-encoder"),
            });

        // ==================================================================
        // Pass 1: Render scene to offscreen texture (notes + keys + particles + hitline)
        // ==================================================================
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: if self.config.bloom_enabled {
                        &gpu.scene_texture_view
                    } else {
                        &final_view
                    },
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Draw piano keys
            pass.set_pipeline(&gpu.key_pipeline);
            pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, gpu.key_instance_buffer.slice(..));
            pass.draw(0..6, 0..key_draw_count as u32);

            // Draw falling notes
            if note_count > 0 {
                pass.set_pipeline(&gpu.note_pipeline);
                pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
                pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, gpu.note_instance_buffer.slice(..));
                pass.draw(0..6, 0..note_count as u32);
            }

            // Draw particles (additive blend)
            if particle_count > 0 {
                pass.set_pipeline(&gpu.particle_pipeline);
                pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
                pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
                pass.set_vertex_buffer(1, gpu.particle_instance_buffer.slice(..));
                pass.draw(0..6, 0..particle_count as u32);
            }
        }

        // Hit-line overlay (separate pass for correct alpha)
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("hitline-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: if self.config.bloom_enabled {
                        &gpu.scene_texture_view
                    } else {
                        &final_view
                    },
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&gpu.hitline_pipeline);
            pass.set_bind_group(0, &gpu.bloom_blur_h_bind_group, &[]);
            pass.set_bind_group(1, &gpu.uniform_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // ==================================================================
        // Pass 2–5: Bloom (only if enabled)
        // ==================================================================
        if self.config.bloom_enabled {
            // Threshold: scene → bloom_a
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom-threshold"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &gpu.bloom_a_texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&gpu.bloom_threshold_pipeline);
                pass.set_bind_group(0, &gpu.bloom_threshold_bind_group, &[]);
                pass.set_bind_group(1, &gpu.uniform_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }

            // Horizontal blur: bloom_a → bloom_b
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom-blur-h"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &gpu.bloom_b_texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&gpu.bloom_blur_h_pipeline);
                pass.set_bind_group(0, &gpu.bloom_blur_h_bind_group, &[]);
                pass.set_bind_group(1, &gpu.uniform_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }

            // Vertical blur: bloom_b → bloom_a
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom-blur-v"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &gpu.bloom_a_texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&gpu.bloom_blur_v_pipeline);
                pass.set_bind_group(0, &gpu.bloom_blur_v_bind_group, &[]);
                pass.set_bind_group(1, &gpu.uniform_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }

            // Composite: bloom_a + scene → final surface
            {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("bloom-composite"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &final_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&gpu.bloom_composite_pipeline);
                pass.set_bind_group(0, &gpu.bloom_composite_bind_group, &[]);
                pass.set_bind_group(1, &gpu.uniform_bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }

        if ui_count > 0 {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ui-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &final_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&gpu.key_pipeline);
            pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, gpu.vertex_buffer.slice(..));
            pass.set_vertex_buffer(1, gpu.ui_instance_buffer.slice(..));
            pass.draw(0..6, 0..ui_count as u32);
        }

        gpu.queue.submit(std::iter::once(enc.finish()));
        output.present();
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

pub fn pitch_to_ndc_x(pitch: u8) -> f32 {
    let white_width = 2.0 / 52.0;
    let mut white_idx = 0u32;
    let mut last_white_x = -1.0 + 0.5 * white_width;

    for midi in PIANO_MIN..=pitch.min(PIANO_MAX) {
        if is_black_key(midi) {
            if midi == pitch {
                return last_white_x + white_width * 0.5;
            }
        } else {
            last_white_x = -1.0 + (white_idx as f32 + 0.5) * white_width;
            if midi == pitch {
                return last_white_x;
            }
            white_idx += 1;
        }
    }
    // Fallback for out-of-range pitches
    -1.0 + 2.0 * (pitch.saturating_sub(PIANO_MIN)) as f32 / PIANO_RANGE
}

pub fn pitch_to_hue(pitch: u8) -> f32 {
    // Register-based palette: bass=warm reds, middle=cyan/blue, treble=violet
    let t = (pitch.saturating_sub(PIANO_MIN)) as f32 / PIANO_RANGE;
    // Map [0,1] to [0°, 300°] so bass starts warm and treble ends cool
    t * 300.0
}

pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let hp = h / 60.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match hp as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (r + m, g + m, b + m)
}
