//! Placeholder model CLI args.

pub struct GemmaArgs {
    pub model_path: String,
    pub tokenizer_path: String,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: Option<u64>,
}

impl Default for GemmaArgs {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            tokenizer_path: String::new(),
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            seed: None,
        }
    }
}
