use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[derive(Debug, Clone, Copy)]
pub enum TransportCommand {
    Play,
    Pause,
    SeekSamples(u64),
}

pub struct TransportState {
    paused: AtomicBool,
    total_samples: AtomicU64,
}

impl Default for TransportState {
    fn default() -> Self {
        Self::new()
    }
}

impl TransportState {
    pub fn new() -> Self {
        Self {
            paused: AtomicBool::new(false),
            total_samples: AtomicU64::new(0),
        }
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Acquire)
    }

    pub fn set_paused(&self, paused: bool) {
        self.paused.store(paused, Ordering::Release);
    }

    pub fn total_samples(&self) -> u64 {
        self.total_samples.load(Ordering::Acquire)
    }

    pub fn set_total_samples(&self, total_samples: u64) {
        self.total_samples.store(total_samples, Ordering::Release);
    }
}
