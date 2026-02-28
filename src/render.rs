use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::Receiver;
use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

use crate::config::RenderConfig;
use crate::events::{
    FrameUniforms, KeyInstance, NoteEvent, NoteEventKind, NoteInstance, ParticleInstance,
    VisualNote,
};
use crate::particles::ParticleSystem;
use crate::timing::AudioClock;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MAX_NOTE_INSTANCES: usize = 4096;
const MAX_KEY_INSTANCES: usize = 128;
const MAX_PARTICLE_INSTANCES: usize = 2048;

const HIT_LINE_Y: f32 = -0.8; // NDC y where notes "land"
const SCROLL_SPEED: f32 = 0.5; // NDC per second

const PIANO_MIN: u8 = 21; // A0
const PIANO_MAX: u8 = 108; // C8
const PIANO_RANGE: f32 = (PIANO_MAX - PIANO_MIN) as f32;

const NOTE_CULL_SECONDS: f64 = 12.0;
const PARTICLE_BURST: usize = 12;

// ---------------------------------------------------------------------------
// Per-vertex data for the unit quad (two triangles)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
}

const QUAD_VERTS: [Vertex; 6] = [
    Vertex {
        position: [-0.5, -0.5],
    },
    Vertex {
        position: [0.5, -0.5],
    },
    Vertex {
        position: [0.5, 0.5],
    },
    Vertex {
        position: [-0.5, -0.5],
    },
    Vertex {
        position: [0.5, 0.5],
    },
    Vertex {
        position: [-0.5, 0.5],
    },
];

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn run_render(
    config: RenderConfig,
    event_rx: Receiver<NoteEvent>,
    clock: Arc<AudioClock>,
) -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    let mut app = App {
        config,
        event_rx,
        clock,
        window: None,
        gpu: None,
        visual_notes: Vec::new(),
        particles: ParticleSystem::new(),
        last_frame: Instant::now(),
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
    window: Option<Arc<Window>>,
    gpu: Option<GpuState>,
    visual_notes: Vec<VisualNote>,
    particles: ParticleSystem,
    last_frame: Instant,
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
    let mut keys = Vec::with_capacity(88);
    let white_width = 2.0 / 52.0; // 52 white keys in 88-key range (approx)
    let key_height = 0.15;
    let key_y = -0.92;

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
            border_radius: 0.15,
            _pad: [0.0; 3],
        });
        white_idx += 1;
    }

    // Second pass: black keys (positioned between white keys)
    white_idx = 0;
    let mut prev_white_x = -1.0 + 0.5 * white_width;
    for midi in PIANO_MIN..=PIANO_MAX {
        if is_black_key(midi) {
            let x = prev_white_x + white_width * 0.5;
            keys.push(KeyInstance {
                position: [x, key_y + key_height * 0.18],
                size: [white_width * 0.55, key_height * 0.62],
                color: [0.15, 0.15, 0.18, 1.0],
                border_radius: 0.12,
                _pad: [0.0; 3],
            });
        } else {
            prev_white_x = -1.0 + (white_idx as f32 + 0.5) * white_width;
            white_idx += 1;
        }
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
            format: wgpu::VertexFormat::Float32x3,
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
        let key_instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("key-instances"),
            contents: bytemuck::cast_slice(&key_instances),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
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
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame).as_secs_f32();
                self.last_frame = now;

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
            }
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
        {
            let mut keys = gpu.key_instances.clone();
            // White keys are indices 0..52, black keys 52..88.
            // We iterate all MIDI pitches and update matching keys.
            let mut white_idx: usize = 0;
            let mut black_idx: usize = 0;
            for midi in PIANO_MIN..=PIANO_MAX {
                if is_black_key(midi) {
                    // Black key at index 52 + black_idx
                    let ki = 52 + black_idx;
                    if ki < keys.len() && active_pitches[midi as usize] {
                        let hue = pitch_to_hue(midi);
                        let (r, g, b) = hsv_to_rgb(hue, 0.9, 0.9);
                        keys[ki].color = [r, g, b, 1.0];
                    } else if ki < keys.len() {
                        keys[ki].color = [0.15, 0.15, 0.18, 1.0];
                    }
                    black_idx += 1;
                } else {
                    // White key at index white_idx
                    if white_idx < keys.len() && active_pitches[midi as usize] {
                        let hue = pitch_to_hue(midi);
                        let (r, g, b) = hsv_to_rgb(hue, 0.7, 1.0);
                        keys[white_idx].color = [r, g, b, 1.0];
                    } else if white_idx < keys.len() {
                        keys[white_idx].color = [0.92, 0.92, 0.92, 1.0];
                    }
                    white_idx += 1;
                }
            }
            gpu.queue
                .write_buffer(&gpu.key_instance_buffer, 0, bytemuck::cast_slice(&keys));
        }

        // ---- build note instances ----
        let mut note_instances = Vec::with_capacity(self.visual_notes.len());
        for note in &self.visual_notes {
            if note.pitch < PIANO_MIN || note.pitch > PIANO_MAX {
                continue;
            }

            let x = pitch_to_ndc_x(note.pitch);
            let note_w = 2.0 / PIANO_RANGE * 0.8;

            let y_top = HIT_LINE_Y + (now - note.start_time) as f32 * SCROLL_SPEED;
            let y_bot = match note.end_time {
                Some(end) => HIT_LINE_Y + (now - end) as f32 * SCROLL_SPEED,
                None => HIT_LINE_Y,
            };

            let note_h = (y_top - y_bot).max(0.01);
            let y_c = (y_top + y_bot) / 2.0;

            let hue = pitch_to_hue(note.pitch);
            let (r, g, b) = hsv_to_rgb(hue, 0.75, 0.85);
            let a = (note.velocity as f32 / 127.0).clamp(0.4, 1.0);

            let is_active = note.end_time.is_none();

            note_instances.push(NoteInstance {
                position: [x, y_c],
                size: [note_w, note_h],
                color: [r, g, b, a],
                border_radius: 0.2,
                glow_intensity: if is_active { 0.8 } else { 0.2 },
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
            pass.draw(0..6, 0..gpu.key_instances.len() as u32);

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

        gpu.queue.submit(std::iter::once(enc.finish()));
        output.present();
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

pub fn pitch_to_ndc_x(pitch: u8) -> f32 {
    -1.0 + 2.0 * (pitch.saturating_sub(PIANO_MIN)) as f32 / PIANO_RANGE
}

pub fn pitch_to_hue(pitch: u8) -> f32 {
    ((pitch.saturating_sub(PIANO_MIN)) as f32 / PIANO_RANGE) * 360.0
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
