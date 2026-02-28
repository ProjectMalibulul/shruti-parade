// ============================================================================
// Shruti Parade — instanced geometry shader
//
// Covers: falling notes (SDF rounded-rect + glow), piano keys, particles.
// Bind group 0: FrameUniforms (resolution, time).
// ============================================================================

struct FrameUniforms {
    resolution: vec2<f32>,
    time: f32,
    _pad: f32,
};
@group(0) @binding(0) var<uniform> frame: FrameUniforms;

// ---- SDF helper ----
fn sdf_rounded_rect(p: vec2<f32>, half_size: vec2<f32>, radius: f32) -> f32 {
    let q = abs(p) - half_size + vec2<f32>(radius);
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

// ===========================================================================
// Notes
// ===========================================================================
struct NoteVsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) local_uv: vec2<f32>,
    @location(2) inst_size: vec2<f32>,
    @location(3) border_radius: f32,
    @location(4) glow_intensity: f32,
};

@vertex
fn vs_notes(
    @location(0) quad_pos: vec2<f32>,
    @location(1) inst_pos: vec2<f32>,
    @location(2) inst_size: vec2<f32>,
    @location(3) inst_color: vec4<f32>,
    @location(4) border_radius: f32,
    @location(5) glow_intensity: f32,
) -> NoteVsOut {
    var out: NoteVsOut;
    let glow_margin = vec2<f32>(1.0 + 0.2 * glow_intensity);
    let world_pos = quad_pos * inst_size * glow_margin + inst_pos;
    out.clip_position = vec4<f32>(world_pos, 0.0, 1.0);
    out.color = inst_color;
    out.local_uv = quad_pos * glow_margin;
    out.inst_size = inst_size * frame.resolution * 0.5;
    out.border_radius = border_radius;
    out.glow_intensity = glow_intensity;
    return out;
}

@fragment
fn fs_notes(in: NoteVsOut) -> @location(0) vec4<f32> {
    let half_size = in.inst_size * 0.5;
    let p = in.local_uv * in.inst_size;
    let r = in.border_radius * min(half_size.x, half_size.y);
    let d = sdf_rounded_rect(p, half_size, r);
    let aa = 1.0 - smoothstep(-1.5, 0.5, d);
    let glow_width = 8.0 * in.glow_intensity;
    let glow = in.glow_intensity * exp(-d * d / (glow_width * glow_width + 0.001)) * 0.5;
    let grad = 1.0 + in.local_uv.y * 0.3;
    var col = in.color;
    col = vec4<f32>(col.rgb * grad, col.a);
    let body = col * aa;
    let glow_col = vec4<f32>(col.rgb * 1.5, glow);
    return body + glow_col * (1.0 - aa);
}

// ===========================================================================
// Piano keys
// ===========================================================================
struct KeyVsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) local_uv: vec2<f32>,
    @location(2) inst_size: vec2<f32>,
    @location(3) border_radius: f32,
};

@vertex
fn vs_keys(
    @location(0) quad_pos: vec2<f32>,
    @location(1) inst_pos: vec2<f32>,
    @location(2) inst_size: vec2<f32>,
    @location(3) inst_color: vec4<f32>,
    @location(4) border_radius: f32,
) -> KeyVsOut {
    var out: KeyVsOut;
    let world_pos = quad_pos * inst_size + inst_pos;
    out.clip_position = vec4<f32>(world_pos, 0.0, 1.0);
    out.color = inst_color;
    out.local_uv = quad_pos;
    out.inst_size = inst_size * frame.resolution * 0.5;
    out.border_radius = border_radius;
    return out;
}

@fragment
fn fs_keys(in: KeyVsOut) -> @location(0) vec4<f32> {
    let half_size = in.inst_size * 0.5;
    let p = in.local_uv * in.inst_size;
    let r = in.border_radius * min(half_size.x, half_size.y);
    let d = sdf_rounded_rect(p, half_size, r);
    let aa = 1.0 - smoothstep(-1.0, 0.5, d);
    let grad = 0.9 + in.local_uv.y * 0.2;
    return vec4<f32>(in.color.rgb * grad, in.color.a * aa);
}

// ===========================================================================
// Particles
// ===========================================================================
struct ParticleVsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) local_uv: vec2<f32>,
};

@vertex
fn vs_particles(
    @location(0) quad_pos: vec2<f32>,
    @location(1) inst_pos: vec2<f32>,
    @location(2) inst_size: vec2<f32>,
    @location(3) inst_color: vec4<f32>,
) -> ParticleVsOut {
    var out: ParticleVsOut;
    let world_pos = quad_pos * inst_size + inst_pos;
    out.clip_position = vec4<f32>(world_pos, 0.0, 1.0);
    out.color = inst_color;
    out.local_uv = quad_pos;
    return out;
}

@fragment
fn fs_particles(in: ParticleVsOut) -> @location(0) vec4<f32> {
    let d = length(in.local_uv) * 2.0;
    let alpha = 1.0 - smoothstep(0.5, 1.0, d);
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
