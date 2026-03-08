// ============================================================================
// Shruti Parade — fullscreen pass shader
//
// Covers: bloom (threshold, H-blur, V-blur, composite), hit-line overlay.
// Bind group 0: bloom_texture + bloom_sampler (+ optional scene_texture).
// Bind group 1: FrameUniforms (resolution, time) — used by blur passes.
// ============================================================================

@group(0) @binding(0) var bloom_texture: texture_2d<f32>;
@group(0) @binding(1) var bloom_sampler: sampler;

struct FrameUniforms {
    resolution: vec2<f32>,
    time: f32,
    _pad: f32,
};
@group(1) @binding(0) var<uniform> frame: FrameUniforms;

struct FullscreenOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) idx: u32) -> FullscreenOutput {
    // Full-screen triangle trick (3 verts, no buffer needed)
    var out: FullscreenOutput;
    let x = f32(i32(idx & 1u)) * 4.0 - 1.0;
    let y = f32(i32(idx >> 1u)) * 4.0 - 1.0;
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

// --- Bloom threshold ---
@fragment
fn fs_bloom_threshold(in: FullscreenOutput) -> @location(0) vec4<f32> {
    let col = textureSampleLevel(bloom_texture, bloom_sampler, in.uv, 0.0);
    let lum = dot(col.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
    let t = smoothstep(0.4, 0.8, lum);
    return vec4<f32>(col.rgb * t, 1.0);
}

// --- Bloom horizontal blur (manually unrolled 9-tap Gaussian) ---
@fragment
fn fs_bloom_blur_h(in: FullscreenOutput) -> @location(0) vec4<f32> {
    let px = 1.0 / frame.resolution.x;
    var r = textureSampleLevel(bloom_texture, bloom_sampler, in.uv, 0.0).rgb * 0.227027;

    let o1 = vec2<f32>(px, 0.0);
    r += textureSampleLevel(bloom_texture, bloom_sampler, in.uv + o1, 0.0).rgb * 0.1945946;
    r += textureSampleLevel(bloom_texture, bloom_sampler, in.uv - o1, 0.0).rgb * 0.1945946;

    let o2 = vec2<f32>(px * 2.0, 0.0);
    r += textureSampleLevel(bloom_texture, bloom_sampler, in.uv + o2, 0.0).rgb * 0.1216216;
    r += textureSampleLevel(bloom_texture, bloom_sampler, in.uv - o2, 0.0).rgb * 0.1216216;

    let o3 = vec2<f32>(px * 3.0, 0.0);
    r += textureSampleLevel(bloom_texture, bloom_sampler, in.uv + o3, 0.0).rgb * 0.054054;
    r += textureSampleLevel(bloom_texture, bloom_sampler, in.uv - o3, 0.0).rgb * 0.054054;

    let o4 = vec2<f32>(px * 4.0, 0.0);
    r += textureSampleLevel(bloom_texture, bloom_sampler, in.uv + o4, 0.0).rgb * 0.016216;
    r += textureSampleLevel(bloom_texture, bloom_sampler, in.uv - o4, 0.0).rgb * 0.016216;

    return vec4<f32>(r, 1.0);
}

// --- Bloom vertical blur (manually unrolled 9-tap Gaussian) ---
@fragment
fn fs_bloom_blur_v(in: FullscreenOutput) -> @location(0) vec4<f32> {
    let py = 1.0 / frame.resolution.y;
    var r = textureSampleLevel(bloom_texture, bloom_sampler, in.uv, 0.0).rgb * 0.227027;

    let o1 = vec2<f32>(0.0, py);
    r += textureSampleLevel(bloom_texture, bloom_sampler, in.uv + o1, 0.0).rgb * 0.1945946;
    r += textureSampleLevel(bloom_texture, bloom_sampler, in.uv - o1, 0.0).rgb * 0.1945946;

    let o2 = vec2<f32>(0.0, py * 2.0);
    r += textureSampleLevel(bloom_texture, bloom_sampler, in.uv + o2, 0.0).rgb * 0.1216216;
    r += textureSampleLevel(bloom_texture, bloom_sampler, in.uv - o2, 0.0).rgb * 0.1216216;

    let o3 = vec2<f32>(0.0, py * 3.0);
    r += textureSampleLevel(bloom_texture, bloom_sampler, in.uv + o3, 0.0).rgb * 0.054054;
    r += textureSampleLevel(bloom_texture, bloom_sampler, in.uv - o3, 0.0).rgb * 0.054054;

    let o4 = vec2<f32>(0.0, py * 4.0);
    r += textureSampleLevel(bloom_texture, bloom_sampler, in.uv + o4, 0.0).rgb * 0.016216;
    r += textureSampleLevel(bloom_texture, bloom_sampler, in.uv - o4, 0.0).rgb * 0.016216;

    return vec4<f32>(r, 1.0);
}

// --- Bloom composite: scene + bloom ---
@group(0) @binding(2) var scene_texture: texture_2d<f32>;

@fragment
fn fs_bloom_composite(in: FullscreenOutput) -> @location(0) vec4<f32> {
    let scene = textureSampleLevel(scene_texture, bloom_sampler, in.uv, 0.0);
    let bloom = textureSampleLevel(bloom_texture, bloom_sampler, in.uv, 0.0);
    return vec4<f32>(scene.rgb + bloom.rgb * 0.6, 1.0);
}

// --- Hit-line overlay ---
// Improved hit-line with subtle rainbow shimmer and wider glow
@fragment
fn fs_hitline(in: FullscreenOutput) -> @location(0) vec4<f32> {
    let line_y = 0.865;
    let dist = abs(in.uv.y - line_y);
    let pulse = 1.0 + 0.12 * sin(frame.time * 2.5);

    // Core line (sharp)
    let core = smoothstep(0.003, 0.0, dist) * pulse;
    // Inner glow (medium)
    let glow_inner = exp(-dist * dist / 0.0004) * 0.45 * pulse;
    // Outer glow (wide, subtle)
    let glow_outer = exp(-dist * dist / 0.003) * 0.12;

    let intensity = core + glow_inner + glow_outer;

    // Subtle horizontal hue shift — rainbow shimmer across the hit-line
    let hue = in.uv.x * 0.8 + frame.time * 0.15;
    let r = 0.50 + 0.10 * sin(hue * 6.28);
    let g = 0.78 + 0.08 * sin(hue * 6.28 + 2.09);
    let b = 1.0;

    return vec4<f32>(r, g, b, intensity);
}
