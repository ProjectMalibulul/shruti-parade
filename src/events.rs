/// Direction of a note event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoteEventKind {
    NoteOn,
    NoteOff,
}

/// A timestamped note event produced by the inference engine.
#[derive(Debug, Clone, Copy)]
pub struct NoteEvent {
    pub kind: NoteEventKind,
    pub pitch: u8,        // MIDI note number 0–127
    pub velocity: u8,     // 0–127
    pub sample_time: u64, // authoritative sample index from AudioClock
}

/// Render-facing note state kept by the visualiser.
#[derive(Debug, Clone, Copy)]
pub struct VisualNote {
    pub pitch: u8,
    pub velocity: u8,
    pub start_time: f64,         // seconds
    pub end_time: Option<f64>,   // None ⇒ still held
}

/// Compact GPU-side per-instance data for a falling note rectangle.
/// Matches the vertex buffer layout declared in the render pipeline.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NoteInstance {
    pub position: [f32; 2],      // centre in NDC coords
    pub size: [f32; 2],          // width, height
    pub color: [f32; 4],         // RGBA
    pub border_radius: f32,      // corner radius in NDC
    pub glow_intensity: f32,     // 0.0 = off, 1.0 = full glow
    pub _pad: [f32; 2],          // align to 48 bytes
}

/// GPU-side per-instance data for a piano key.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct KeyInstance {
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub color: [f32; 4],
    pub border_radius: f32,
    pub _pad: [f32; 3],
}

/// Simple CPU-side particle for impact effects.
#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub color: [f32; 4],
    pub life: f32,       // 0..1, decreases each frame
    pub size: f32,
}

/// GPU-side particle instance data.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleInstance {
    pub position: [f32; 2],
    pub size: [f32; 2],
    pub color: [f32; 4],
}

/// Uniform data pushed per-frame to the GPU.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FrameUniforms {
    pub resolution: [f32; 2],
    pub time: f32,
    pub _pad: f32,
}

