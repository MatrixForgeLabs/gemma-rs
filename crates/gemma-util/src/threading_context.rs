//! Simplified threading context with a shared thread pool.

use crate::allocator::{Allocator, CacheInfo};
use crate::basics::Tristate;
use crate::threading::ThreadPool;
use crate::topology::{BoundedSlice, BoundedTopology};

#[derive(Clone, Debug, Default)]
pub struct ThreadingArgs {
    pub skip_packages: usize,
    pub max_packages: usize,
    pub skip_clusters: usize,
    pub max_clusters: usize,
    pub skip_lps: usize,
    pub max_lps: usize,
    pub bind: Tristate,
    pub max_threads: usize,
    pub pin: Tristate,
    pub spin: Tristate,
}

impl ThreadingArgs {
    pub fn validate(&self) -> Option<&'static str> {
        if self.max_packages != 0 && self.max_packages < 1 {
            return Some("max_packages must be >= 1");
        }
        None
    }
}

pub struct ThreadingContext {
    pub topology: BoundedTopology,
    pub cache_info: CacheInfo,
    pub allocator: Allocator,
    pub pool: ThreadPool,
}

impl ThreadingContext {
    pub fn new(args: &ThreadingArgs) -> Self {
        let topology = BoundedTopology::new(
            BoundedSlice::new(args.skip_packages, args.max_packages),
            BoundedSlice::new(args.skip_clusters, args.max_clusters),
        );
        let cache_info = CacheInfo::new(&topology);
        let allocator = Allocator::new(&topology, &cache_info, args.bind != Tristate::False);

        let pool = if args.max_threads != 0 {
            ThreadPool::new(args.max_threads)
        } else {
            ThreadPool::for_num_cpus()
        };

        Self {
            topology,
            cache_info,
            allocator,
            pool,
        }
    }

    pub fn worker(&self, cluster_idx: usize) -> usize {
        cluster_idx
    }
}
