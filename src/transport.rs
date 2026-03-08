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
    /// Set by feed threads on seek; cleared by output callback after draining.
    flush: AtomicBool,
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
            flush: AtomicBool::new(false),
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

    /// Signal the output callback to drain stale data from the ring.
    pub fn request_flush(&self) {
        self.flush.store(true, Ordering::Release);
    }

    /// Check whether a flush is pending (called from the output callback).
    pub fn should_flush(&self) -> bool {
        self.flush.load(Ordering::Acquire)
    }

    /// Clear the flush flag after draining (called from the output callback).
    pub fn clear_flush(&self) {
        self.flush.store(false, Ordering::Release);
    }
}
