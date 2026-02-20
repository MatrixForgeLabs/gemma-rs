//! Lightweight parallel utilities shared across crates.

use rayon::prelude::*;
use rayon::ThreadPool as RayonPool;
use rayon::ThreadPoolBuilder;

#[derive(Copy, Clone, Debug)]
pub enum ParallelismStrategy {
    None,
    Flat,
}

pub struct ThreadPool {
    pool: RayonPool,
}

impl ThreadPool {
    pub fn new(num_threads: usize) -> Self {
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("failed to build thread pool");
        Self { pool }
    }

    pub fn for_num_cpus() -> Self {
        Self::new(num_cpus::get())
    }

    pub fn install<F: FnOnce() -> R + Send, R: Send>(&self, f: F) -> R {
        self.pool.install(f)
    }
}

pub fn parallel_for<F>(strategy: ParallelismStrategy, num_tasks: usize, pool: &ThreadPool, func: F)
where
    F: Fn(usize, usize) + Send + Sync,
{
    match strategy {
        ParallelismStrategy::None => {
            for task in 0..num_tasks {
                func(task, 0);
            }
        }
        ParallelismStrategy::Flat => {
            pool.install(|| {
                (0..num_tasks).into_par_iter().for_each(|task| {
                    let worker = rayon::current_thread_index().unwrap_or(0);
                    func(task, worker);
                });
            });
        }
    }
}
