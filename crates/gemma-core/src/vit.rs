//! Minimal ViT placeholder.

pub struct Vit {
    pub hidden: usize,
}

impl Vit {
    pub fn new(hidden: usize) -> Self {
        Self { hidden }
    }
}
