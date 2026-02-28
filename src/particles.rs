use crate::events::{Particle, ParticleInstance};

const MAX_PARTICLES: usize = 2048;

/// Simple CPU-side particle pool for note-impact effects.
///
/// On each note impact, `spawn()` ejects a burst of particles from the hit
/// position.  `tick()` advances physics (gravity + drag) and culls dead
/// particles.  `instances()` produces the GPU upload slice.
pub struct ParticleSystem {
    particles: Vec<Particle>,
}

impl ParticleSystem {
    pub fn new() -> Self {
        Self {
            particles: Vec::with_capacity(MAX_PARTICLES),
        }
    }

    /// Spawn a burst of particles at `pos` with the given base `color`.
    pub fn spawn(&mut self, pos: [f32; 2], color: [f32; 4], count: usize) {
        let budget = (MAX_PARTICLES - self.particles.len()).min(count);
        // Simple deterministic pseudo-random: golden-angle spread.
        let golden = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
        for i in 0..budget {
            let angle = golden * i as f32 + pos[0] * 137.0; // seeded by x
            let speed = 0.3 + 0.5 * ((i as f32 * 0.618) % 1.0);
            let vx = angle.cos() * speed;
            let vy = angle.sin().abs() * speed + 0.2; // bias upward
            self.particles.push(Particle {
                position: pos,
                velocity: [vx, vy],
                color,
                life: 1.0,
                size: 0.008 + 0.006 * ((i as f32 * 0.381) % 1.0),
            });
        }
    }

    /// Advance simulation by `dt` seconds.
    pub fn tick(&mut self, dt: f32) {
        let gravity = -1.2_f32;
        let drag = 0.97_f32;

        for p in &mut self.particles {
            p.velocity[0] *= drag;
            p.velocity[1] *= drag;
            p.velocity[1] += gravity * dt;
            p.position[0] += p.velocity[0] * dt;
            p.position[1] += p.velocity[1] * dt;
            p.life -= dt * 1.5; // ~0.67s total lifetime
        }

        self.particles.retain(|p| p.life > 0.0);
    }

    /// Generate GPU instance data for alive particles.
    pub fn instances(&self) -> Vec<ParticleInstance> {
        self.particles
            .iter()
            .map(|p| ParticleInstance {
                position: p.position,
                size: [p.size, p.size],
                color: [
                    p.color[0],
                    p.color[1],
                    p.color[2],
                    p.color[3] * p.life.clamp(0.0, 1.0),
                ],
            })
            .collect()
    }

    pub fn count(&self) -> usize {
        self.particles.len()
    }
}
