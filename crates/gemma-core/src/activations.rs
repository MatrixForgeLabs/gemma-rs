//! Placeholder activations container.

pub struct Activations {
    pub batch: usize,
}

impl Activations {
    pub fn new(batch: usize) -> Self {
        Self { batch }
    }
}
