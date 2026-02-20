//! Simplified topology model.

#[derive(Copy, Clone, Debug, Default)]
pub struct BoundedSlice {
    skip: usize,
    max: usize,
}

impl BoundedSlice {
    pub fn new(skip: usize, max: usize) -> Self {
        Self { skip, max }
    }

    pub fn begin(&self) -> usize {
        self.skip
    }

    pub fn max(&self) -> usize {
        self.max
    }

    pub fn end(&self, detected: usize) -> usize {
        if self.max == 0 {
            detected
        } else {
            (self.skip + self.max).min(detected)
        }
    }

    pub fn num(&self, detected: usize) -> usize {
        self.end(detected).saturating_sub(self.begin())
    }

    pub fn contains(&self, detected: usize, idx: usize) -> bool {
        self.begin() <= idx && idx < self.end(detected)
    }
}

#[derive(Clone, Debug)]
pub struct Cluster {
    num_workers: usize,
    node: usize,
    private_kib: usize,
    shared_kib: usize,
}

impl Cluster {
    pub fn new(num_workers: usize) -> Self {
        Self {
            num_workers,
            node: 0,
            private_kib: 256,
            shared_kib: 1024,
        }
    }

    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    pub fn node(&self) -> usize {
        self.node
    }

    pub fn private_kib(&self) -> usize {
        self.private_kib
    }

    pub fn shared_kib(&self) -> usize {
        self.shared_kib
    }
}

#[derive(Clone, Debug)]
pub struct BoundedTopology {
    clusters: Vec<Cluster>,
    package_slice: BoundedSlice,
    cluster_slice: BoundedSlice,
}

impl BoundedTopology {
    pub fn new(package_slice: BoundedSlice, cluster_slice: BoundedSlice) -> Self {
        let _ = package_slice;
        let _ = cluster_slice;
        Self {
            clusters: vec![Cluster::new(1)],
            package_slice,
            cluster_slice,
        }
    }

    pub fn num_nodes(&self) -> usize {
        1
    }

    pub fn num_clusters(&self) -> usize {
        self.clusters.len()
    }

    pub fn cluster(&self, idx: usize) -> &Cluster {
        &self.clusters[idx]
    }

    pub fn skipped_packages(&self) -> usize {
        self.package_slice.begin()
    }

    pub fn skipped_clusters(&self) -> usize {
        self.cluster_slice.begin()
    }
}
