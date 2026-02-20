//! Lightweight profiling hooks for perf binaries and tests.

use std::time::{Duration, Instant};

pub struct BenchTimer {
    start: Instant,
}

impl BenchTimer {
    pub fn start() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

pub fn ns_per_iter(elapsed: Duration, iters: usize) -> f64 {
    if iters == 0 {
        return 0.0;
    }
    elapsed.as_nanos() as f64 / iters as f64
}
