use std::sync::atomic::{AtomicU64, Ordering};

/// Authoritative audio clock driven by the sample counter in the audio callback.
///
/// The audio callback thread calls `advance()` (Release store).
/// All other threads call the read methods (Acquire load).
/// This is the single source of truth for wall-clock time in the engine.
pub struct AudioClock {
    sample_rate: f64,
    sample_count: AtomicU64,
}

impl AudioClock {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate: sample_rate as f64,
            sample_count: AtomicU64::new(0),
        }
    }

    /// Called exclusively from the audio callback. Wait-free.
    #[inline]
    pub fn advance(&self, frames: u64) {
        self.sample_count.fetch_add(frames, Ordering::Release);
    }

    /// Current time in seconds (monotonic). Safe from any thread.
    #[inline]
    pub fn now_seconds(&self) -> f64 {
        let s = self.sample_count.load(Ordering::Acquire);
        s as f64 / self.sample_rate
    }

    /// Raw sample count.
    #[inline]
    #[allow(dead_code)]
    pub fn now_samples(&self) -> u64 {
        self.sample_count.load(Ordering::Acquire)
    }

    /// Force the clock to a specific sample index (used for seek).
    /// Uses SeqCst to ensure all threads see the new value immediately.
    #[inline]
    pub fn set_samples(&self, sample: u64) {
        self.sample_count.store(sample, Ordering::SeqCst);
    }

    /// Convert a sample index to seconds.
    #[inline]
    pub fn sample_to_seconds(&self, sample: u64) -> f64 {
        sample as f64 / self.sample_rate
    }

    #[allow(dead_code)]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }
}
