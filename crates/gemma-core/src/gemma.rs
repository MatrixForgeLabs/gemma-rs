//! Minimal model stub.

use gemma_io::io::Path;
use gemma_util::allocator::{Allocator, CacheInfo};
use gemma_util::mat::{MatOwner, MatPadding, MatPtr};
use gemma_util::topology::{BoundedSlice, BoundedTopology};

use crate::attention::{build_attention_plan, kv_head_for_query_head};
use crate::configs::{ModelConfig, PostNormType, PromptWrapping, QueryScaleType};
use crate::flash_attention::{run_head_flash_attention, HeadFlashParams};
use crate::kv_cache::KVCache;
use crate::model_store::ModelStore;
use crate::tokenizer::Tokenizer;
use gemma_compression::int_format;
use gemma_compression::nuq;
use gemma_compression::sfp;
use gemma_compression::types::{I8Stream, NuqStream, Type};
use half::bf16;

pub struct Gemma {
    pub config: ModelConfig,
    pub tokenizer: Tokenizer,
    weights: Option<MinimalWeights>,
    pub(crate) full: Option<FullWeights>,
}

#[derive(Copy, Clone, Debug)]
pub struct SamplingOptions {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: Option<u64>,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            seed: None,
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
struct GenerationOptions {
    prefix_end: Option<usize>,
    sampling: SamplingOptions,
}

struct DecodeScratch {
    norm_scale: Vec<f32>,
    logits: Vec<f32>,
    tmp_row: Vec<f32>,
}

impl DecodeScratch {
    fn new(full: &FullWeights) -> Self {
        Self {
            norm_scale: vec![0.0; full.model_dim],
            logits: vec![0.0; full.embedding.rows()],
            tmp_row: vec![0.0; full.model_dim],
        }
    }
}

pub(crate) struct SamplingState {
    opts: SamplingOptions,
    rng: Rng64,
}

impl SamplingState {
    pub(crate) fn new(opts: SamplingOptions) -> Self {
        let seed = opts.seed.unwrap_or(0x9E3779B97F4A7C15);
        Self {
            opts,
            rng: Rng64::new(seed),
        }
    }
}

struct Rng64 {
    state: u64,
}

impl Rng64 {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 { 0xA5A5A5A5A5A5A5A5 } else { seed };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        let v = (self.next_u64() >> 40) as u32;
        v as f32 / ((1u32 << 24) as f32)
    }
}

struct MinimalWeights {
    embedding: MatPtr,
    #[allow(dead_code)]
    _embedding_owner: MatOwner,
    final_norm: Option<(MatPtr, MatOwner)>,
}

pub(crate) struct LayerWeights {
    pub(crate) qkv_ein: MatPtr,
    #[allow(dead_code)]
    pub(crate) qkv_owner: MatOwner,
    pub(crate) att_ein: MatPtr,
    #[allow(dead_code)]
    pub(crate) att_owner: MatOwner,
    pub(crate) gating_ein: MatPtr,
    #[allow(dead_code)]
    pub(crate) gating_owner: MatOwner,
    pub(crate) linear_w: MatPtr,
    #[allow(dead_code)]
    pub(crate) linear_owner: MatOwner,
    pub(crate) pre_att_ns: MatPtr,
    #[allow(dead_code)]
    pub(crate) pre_att_owner: MatOwner,
    pub(crate) pre_ff_ns: MatPtr,
    #[allow(dead_code)]
    pub(crate) pre_ff_owner: MatOwner,
    pub(crate) post_att_ns: MatPtr,
    #[allow(dead_code)]
    pub(crate) post_att_owner: MatOwner,
    pub(crate) post_ff_ns: MatPtr,
    #[allow(dead_code)]
    pub(crate) post_ff_owner: MatOwner,
    pub(crate) key_norm_scale: Option<(MatPtr, MatOwner)>,
    pub(crate) query_norm_scale: Option<(MatPtr, MatOwner)>,
}

pub(crate) struct FullWeights {
    pub(crate) embedding: MatPtr,
    #[allow(dead_code)]
    pub(crate) _embedding_owner: MatOwner,
    pub(crate) final_norm: MatPtr,
    #[allow(dead_code)]
    pub(crate) _final_norm_owner: MatOwner,
    pub(crate) layers: Vec<LayerWeights>,
    pub(crate) model_dim: usize,
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) qkv_dim: usize,
    pub(crate) ff_hidden_dim: usize,
    pub(crate) max_seq_len: usize,
    pub(crate) attention_window_sizes: Vec<usize>,
    pub(crate) att_cap: f32,
    pub(crate) query_scale: QueryScaleType,
    pub(crate) layer_configs: Vec<crate::configs::LayerConfig>,
}

/// Mutable decoder state that keeps KV cache across chat turns.
pub struct CacheState {
    pub cache: Vec<KVCache>,
    pub pos: usize,
}

impl CacheState {
    fn new(full: &FullWeights) -> Self {
        Self {
            cache: (0..full.layers.len())
                .map(|_| KVCache::new(full.max_seq_len, full.kv_heads, full.qkv_dim))
                .collect(),
            pos: 0,
        }
    }
}

#[derive(Copy, Clone)]
struct LayerDebugCtx {
    phase: &'static str,
    pos: usize,
    step: usize,
}

impl Gemma {
    pub fn new(config: ModelConfig) -> Self {
        let tokenizer = Tokenizer::from_bytes(b"unavailable").expect("tokenizer blob required");
        Self {
            config,
            tokenizer,
            weights: None,
            full: None,
        }
    }

    pub fn with_tokenizer(config: ModelConfig, tokenizer: Tokenizer) -> Self {
        Self {
            config,
            tokenizer,
            weights: None,
            full: None,
        }
    }

    pub fn from_sbs(config: ModelConfig, weights_path: &str) -> Self {
        let store = ModelStore::new(Path::new(weights_path));
        let bytes = store
            .read_blob("tokenizer")
            .expect("tokenizer blob missing");
        let tokenizer = Tokenizer::from_bytes(&bytes).expect("failed to load tokenizer blob");
        let config = store.config().cloned().unwrap_or(config);
        let weights = load_minimal_weights(&store);
        let full = load_full_weights(&store, &config);
        Self {
            config,
            tokenizer,
            weights,
            full,
        }
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize) -> String {
        self.generate_with_options(
            prompt,
            max_tokens,
            GenerationOptions {
                prefix_end: None,
                sampling: SamplingOptions::default(),
            },
        )
    }

    pub fn generate_with_sampling(
        &self,
        prompt: &str,
        max_tokens: usize,
        sampling: SamplingOptions,
    ) -> String {
        self.generate_with_options(
            prompt,
            max_tokens,
            GenerationOptions {
                prefix_end: None,
                sampling,
            },
        )
    }

    /// Generate from a pre-formatted conversation string.
    ///
    /// The caller is responsible for building the full conversation with
    /// BOS token and turn markers (e.g. `<start_of_turn>user\n...<end_of_turn>\n`).
    /// This method tokenizes the string directly without calling `wrap_tokens`.
    pub fn generate_chat(
        &self,
        formatted_conversation: &str,
        max_tokens: usize,
        sampling: SamplingOptions,
    ) -> String {
        let mut tokens = self.tokenizer.encode(formatted_conversation);
        if tokens.is_empty() || max_tokens == 0 {
            return String::new();
        }
        // Prepend BOS token — wrap_tokens normally does this but we bypass it.
        const BOS_ID: i32 = 2;
        tokens.insert(0, BOS_ID);
        let prompt_len = tokens.len();
        let mut samp = SamplingState::new(sampling);
        let out_tokens = if let Some(full) = &self.full {
            generate_full_tokens(
                self,
                full,
                &tokens,
                max_tokens,
                GenerationOptions {
                    prefix_end: None,
                    sampling,
                },
                &mut samp,
            )
        } else {
            return String::new();
        };
        let generated = if out_tokens.len() > prompt_len {
            self.tokenizer.decode(&out_tokens[prompt_len..])
        } else {
            String::new()
        };
        strip_terminal_turn_markers(generated)
    }

    /// Initialize a reusable KV cache state for incremental chat decoding.
    pub fn init_cache_state(&self) -> Option<CacheState> {
        self.full.as_ref().map(CacheState::new)
    }

    /// Generate a chat response using an existing KV cache.
    ///
    /// `chunk_tokens` should contain only the newly appended conversation
    /// segment for this turn (including markers). Insert the BOS token in
    /// `chunk_tokens` only on the first call.
    pub fn generate_chat_with_cache(
        &self,
        state: &mut CacheState,
        chunk_tokens: &[i32],
        max_tokens: usize,
        sampling: SamplingOptions,
    ) -> Option<String> {
        let full = self.full.as_ref()?;
        let mut samp = SamplingState::new(sampling);
        let out_tokens = generate_full_tokens_with_cache(
            self,
            full,
            chunk_tokens,
            max_tokens,
            GenerationOptions {
                prefix_end: None,
                sampling,
            },
            &mut samp,
            state.pos,
            Some(&mut state.cache),
            None,
        );
        state.pos += out_tokens.len();
        let generated = if out_tokens.len() > chunk_tokens.len() {
            self.tokenizer.decode(&out_tokens[chunk_tokens.len()..])
        } else {
            String::new()
        };
        Some(strip_terminal_turn_markers(generated))
    }

    /// Streaming variant of `generate_chat_with_cache`. Calls `on_text` as soon
    /// as each non-stop token is produced.
    pub fn generate_chat_streaming_with_cache(
        &self,
        state: &mut CacheState,
        chunk_tokens: &[i32],
        max_tokens: usize,
        sampling: SamplingOptions,
        mut on_text: impl FnMut(&str),
    ) -> Option<String> {
        let full = self.full.as_ref()?;
        let mut samp = SamplingState::new(sampling);
        let mut assembled = String::new();
        let mut stream_cb = |tok: i32| {
            if should_stop_token(tok, &self.config) {
                return;
            }
            let piece = self.tokenizer.decode(&[tok]);
            if !piece.is_empty() {
                on_text(&piece);
                assembled.push_str(&piece);
            }
        };
        let mut cb_ref: &mut dyn FnMut(i32) = &mut stream_cb;
        let out_tokens = generate_full_tokens_with_cache(
            self,
            full,
            chunk_tokens,
            max_tokens,
            GenerationOptions {
                prefix_end: None,
                sampling,
            },
            &mut samp,
            state.pos,
            Some(&mut state.cache),
            Some(&mut cb_ref),
        );
        state.pos += out_tokens.len();
        Some(strip_terminal_turn_markers(assembled))
    }

    pub fn generate_with_prefix_end(
        &self,
        prompt: &str,
        max_tokens: usize,
        prefix_end: Option<usize>,
    ) -> String {
        self.generate_with_options(
            prompt,
            max_tokens,
            GenerationOptions {
                prefix_end,
                sampling: SamplingOptions::default(),
            },
        )
    }

    pub fn has_minimal_weights(&self) -> bool {
        self.weights.is_some()
    }

    pub fn has_full_weights(&self) -> bool {
        self.full.is_some()
    }

    fn generate_with_options(
        &self,
        prompt: &str,
        max_tokens: usize,
        opts: GenerationOptions,
    ) -> String {
        let prompt_tokens = self.tokenizer.encode(prompt);
        let tokens = self.wrap_tokens(prompt_tokens, 0);
        if tokens.is_empty() {
            return prompt.to_string();
        }
        if max_tokens == 0 {
            return prompt.to_string();
        }
        let mut sampling = SamplingState::new(opts.sampling);
        let out_tokens = if let Some(full) = &self.full {
            generate_full_tokens(self, full, &tokens, max_tokens, opts, &mut sampling)
        } else if let Some(weights) = &self.weights {
            generate_minimal_tokens(self, weights, &tokens, max_tokens, &mut sampling)
        } else {
            return format!("{prompt} [stub]");
        };
        let generated = if out_tokens.len() > tokens.len() {
            self.tokenizer.decode(&out_tokens[tokens.len()..])
        } else {
            String::new()
        };
        strip_terminal_turn_markers(generated)
    }

    pub(crate) fn wrap_tokens(&self, mut tokens: Vec<i32>, pos: usize) -> Vec<i32> {
        const BOS_ID: i32 = 2;
        match self.config.wrapping {
            PromptWrapping::GemmaIt | PromptWrapping::GemmaVlm => {
                let sot_user = self.tokenizer.encode("<start_of_turn>user\n");
                let sot_model = self.tokenizer.encode("<start_of_turn>model\n");
                let eot = self.tokenizer.encode("<end_of_turn>\n");
                let mut out = Vec::with_capacity(
                    1 + sot_user.len() + tokens.len() + eot.len() + sot_model.len(),
                );
                if pos == 0 {
                    out.push(BOS_ID);
                } else {
                    out.extend_from_slice(&eot);
                }
                out.extend_from_slice(&sot_user);
                out.append(&mut tokens);
                out.extend_from_slice(&eot);
                out.extend_from_slice(&sot_model);
                out
            }
            _ => {
                if pos == 0 {
                    tokens.insert(0, BOS_ID);
                }
                tokens
            }
        }
    }

    #[allow(dead_code)]
    fn wrap_and_tokenize(&self, prompt: &str, pos: usize) -> Vec<i32> {
        self.wrap_tokens(self.tokenizer.encode(prompt), pos)
    }
}

fn load_minimal_weights(store: &ModelStore) -> Option<MinimalWeights> {
    let embedding = store.mat_ptr("c_embedding")?;
    let final_norm = store.mat_ptr("c_final_norm");

    let topo = BoundedTopology::new(BoundedSlice::new(0, 0), BoundedSlice::new(0, 0));
    let cache = CacheInfo::new(&topo);
    let allocator = Allocator::new(&topo, &cache, false);

    let (embedding, embedding_owner) = load_mat(store, embedding, &allocator).ok()?;
    let final_norm = final_norm.and_then(|mat| load_mat(store, mat, &allocator).ok());

    Some(MinimalWeights {
        embedding,
        _embedding_owner: embedding_owner,
        final_norm,
    })
}

fn load_mat(
    store: &ModelStore,
    mut mat: MatPtr,
    allocator: &Allocator,
) -> Result<(MatPtr, MatOwner), ()> {
    let mut owner = MatOwner::new();
    owner.allocate_for(&mut mat, allocator, MatPadding::Packed);
    let range = store
        .find_and_update_mat_ptr(&mut mat, Type::Unknown)
        .ok_or(())?;
    let bytes = mat.packed_bytes();
    let ok = store.read_range_into(&range, unsafe {
        std::slice::from_raw_parts_mut(mat.row_bytes(0), bytes)
    });
    if !ok {
        return Err(());
    }
    Ok((mat, owner))
}

pub(crate) fn read_row_f32(mat: &MatPtr, row: usize, out: &mut [f32]) {
    assert_eq!(out.len(), mat.cols());
    let cols = mat.cols();
    let row_ofs = row * cols;
    match mat.ty() {
        Type::F32 => unsafe {
            let ptr = mat.row_bytes(row) as *const f32;
            let src = std::slice::from_raw_parts(ptr, cols);
            out.copy_from_slice(src);
        },
        Type::BF16 => unsafe {
            let ptr = mat.row_bytes(row) as *const bf16;
            let src = std::slice::from_raw_parts(ptr, cols);
            for (dst, v) in out.iter_mut().zip(src.iter()) {
                *dst = f32::from(*v);
            }
        },
        Type::SFP => unsafe {
            let ptr = mat.row_bytes(0) as *const u8;
            let src = std::slice::from_raw_parts(ptr, mat.packed_bytes());
            let start = row_ofs;
            let end = start + cols;
            sfp::decode_f32(&src[start..end], out);
        },
        Type::NUQ => unsafe {
            let ptr = mat.row_bytes(0) as *const NuqStream;
            let packed = std::slice::from_raw_parts(ptr, mat.packed_bytes());
            nuq::decompress_and_zero_pad_f32(packed, row_ofs, out, cols);
        },
        Type::I8 => unsafe {
            let ptr = mat.row_bytes(0) as *const I8Stream;
            let packed = std::slice::from_raw_parts(ptr, mat.packed_bytes());
            int_format::decompress_and_zero_pad_f32(packed, row_ofs, out, cols);
        },
        _ => {
            out.fill(0.0);
        }
    }
}

fn load_full_weights(store: &ModelStore, config: &ModelConfig) -> Option<FullWeights> {
    if config.model_dim == 0 || config.layers.is_empty() {
        return None;
    }
    let topo = BoundedTopology::new(BoundedSlice::new(0, 0), BoundedSlice::new(0, 0));
    let cache_info = CacheInfo::new(&topo);
    let allocator = Allocator::new(&topo, &cache_info, false);

    let embedding = store.mat_ptr("c_embedding")?;
    let final_norm = store.mat_ptr("c_final_norm")?;

    let (embedding, embedding_owner) = load_mat(store, embedding, &allocator).ok()?;
    let (final_norm, final_owner) = load_mat(store, final_norm, &allocator).ok()?;

    let mut layers = Vec::with_capacity(config.layers.len());
    for idx in 0..config.layers.len() {
        let suffix = format!("_{}", idx);
        let qkv_ein = store.mat_ptr(&format!("qkv_ein{suffix}"))?;
        let att_ein = store.mat_ptr(&format!("att_ein{suffix}"))?;
        let gating_ein = store.mat_ptr(&format!("gating_ein{suffix}"))?;
        let linear_w = store.mat_ptr(&format!("linear_w{suffix}"))?;
        let pre_att_ns = store.mat_ptr(&format!("pre_att_ns{suffix}"))?;
        let pre_ff_ns = store.mat_ptr(&format!("pre_ff_ns{suffix}"))?;
        let post_att_ns = store.mat_ptr(&format!("post_att_ns{suffix}"))?;
        let post_ff_ns = store.mat_ptr(&format!("post_ff_ns{suffix}"))?;

        let (qkv_ein, qkv_owner) = load_mat(store, qkv_ein, &allocator).ok()?;
        let (att_ein, att_owner) = load_mat(store, att_ein, &allocator).ok()?;
        let (gating_ein, gating_owner) = load_mat(store, gating_ein, &allocator).ok()?;
        let (linear_w, linear_owner) = load_mat(store, linear_w, &allocator).ok()?;
        let (pre_att_ns, pre_att_owner) = load_mat(store, pre_att_ns, &allocator).ok()?;
        let (pre_ff_ns, pre_ff_owner) = load_mat(store, pre_ff_ns, &allocator).ok()?;
        let (post_att_ns, post_att_owner) = load_mat(store, post_att_ns, &allocator).ok()?;
        let (post_ff_ns, post_ff_owner) = load_mat(store, post_ff_ns, &allocator).ok()?;

        layers.push(LayerWeights {
            qkv_ein,
            qkv_owner,
            att_ein,
            att_owner,
            gating_ein,
            gating_owner,
            linear_w,
            linear_owner,
            pre_att_ns,
            pre_att_owner,
            pre_ff_ns,
            pre_ff_owner,
            post_att_ns,
            post_att_owner,
            post_ff_ns,
            post_ff_owner,
            key_norm_scale: if config.layers[idx].use_qk_norm {
                store
                    .mat_ptr(&format!("key_norm{suffix}"))
                    .and_then(|m| load_mat(store, m, &allocator).ok())
            } else {
                None
            },
            query_norm_scale: if config.layers[idx].use_qk_norm {
                store
                    .mat_ptr(&format!("query_norm{suffix}"))
                    .and_then(|m| load_mat(store, m, &allocator).ok())
            } else {
                None
            },
        });
    }

    let layer0 = &config.layers[0];
    Some(FullWeights {
        embedding,
        _embedding_owner: embedding_owner,
        final_norm,
        _final_norm_owner: final_owner,
        layers,
        model_dim: config.model_dim as usize,
        heads: layer0.heads as usize,
        kv_heads: layer0.kv_heads as usize,
        qkv_dim: layer0.qkv_dim as usize,
        ff_hidden_dim: layer0.ff_hidden_dim as usize,
        max_seq_len: config.max_seq_len as usize,
        attention_window_sizes: config
            .attention_window_sizes
            .iter()
            .map(|&v| v as usize)
            .collect(),
        att_cap: config.att_cap,
        query_scale: config.query_scale,
        layer_configs: config.layers.clone(),
    })
}

#[allow(dead_code)]
fn generate_minimal(
    model: &Gemma,
    weights: &MinimalWeights,
    tokens: &[i32],
    max_tokens: usize,
    sampling: &mut SamplingState,
) -> String {
    let out_tokens = generate_minimal_tokens(model, weights, tokens, max_tokens, sampling);
    model.tokenizer.decode(&out_tokens)
}

fn generate_minimal_tokens(
    model: &Gemma,
    weights: &MinimalWeights,
    tokens: &[i32],
    max_tokens: usize,
    sampling: &mut SamplingState,
) -> Vec<i32> {
    let vocab = weights.embedding.rows();
    let dim = weights.embedding.cols();
    let mut out_tokens = tokens.to_vec();

    for _ in 0..max_tokens {
        let last_id = match out_tokens.last() {
            Some(v) => *v as usize,
            None => break,
        };
        if last_id >= vocab {
            break;
        }

        let mut hidden = vec![0.0f32; dim];
        read_row_f32(&weights.embedding, last_id, &mut hidden);

        if let Some((norm, _owner)) = &weights.final_norm {
            if norm.cols() == dim {
                let mut scale = vec![0.0f32; dim];
                read_row_f32(norm, 0, &mut scale);
                for i in 0..dim {
                    hidden[i] *= scale[i];
                }
            }
        }

        let mut tmp_row = vec![0.0f32; dim];
        let mut logits = vec![0.0f32; vocab];
        for row in 0..vocab {
            read_row_f32(&weights.embedding, row, &mut tmp_row);
            let mut sum = 0.0f32;
            for i in 0..dim {
                sum += hidden[i] * tmp_row[i];
            }
            logits[row] = soft_cap(sum, model.config.final_cap);
        }

        let next = sample_token(&mut logits, sampling) as i32;
        out_tokens.push(next);
        if should_stop_token(next, &model.config) {
            break;
        }
    }
    out_tokens
}

#[allow(dead_code)]
fn generate_full(
    model: &Gemma,
    full: &FullWeights,
    tokens: &[i32],
    max_tokens: usize,
    opts: GenerationOptions,
    sampling: &mut SamplingState,
) -> String {
    let out_tokens = generate_full_tokens(model, full, tokens, max_tokens, opts, sampling);
    model.tokenizer.decode(&out_tokens)
}

fn generate_full_tokens(
    model: &Gemma,
    full: &FullWeights,
    tokens: &[i32],
    max_tokens: usize,
    opts: GenerationOptions,
    sampling: &mut SamplingState,
) -> Vec<i32> {
    generate_full_tokens_with_cache(
        model, full, tokens, max_tokens, opts, sampling, 0, None, None,
    )
}

fn generate_full_tokens_with_cache(
    model: &Gemma,
    full: &FullWeights,
    prompt_tokens: &[i32],
    max_tokens: usize,
    opts: GenerationOptions,
    sampling: &mut SamplingState,
    start_pos: usize,
    cache: Option<&mut [KVCache]>,
    mut on_token: Option<&mut dyn FnMut(i32)>,
) -> Vec<i32> {
    let mut owned_cache;
    let cache = if let Some(c) = cache {
        c
    } else {
        owned_cache = (0..full.layers.len())
            .map(|_| KVCache::new(full.max_seq_len, full.kv_heads, full.qkv_dim))
            .collect::<Vec<_>>();
        &mut owned_cache[..]
    };

    let mut hidden = vec![0.0f32; full.model_dim];
    let mut scratch = DecodeScratch::new(full);
    let mut out_tokens = prompt_tokens.to_vec();

    let debug_layer_stats = std::env::var("GEMMA_DEBUG_LAYER_STATS")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    prefill_full_prompt(
        full,
        prompt_tokens,
        cache,
        &mut hidden,
        opts.prefix_end,
        start_pos,
    );

    let debug_topk = std::env::var("GEMMA_DEBUG_TOPK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    for step in 0..max_tokens {
        let next = decode_full_next_token(
            model,
            full,
            cache,
            &mut hidden,
            &out_tokens,
            opts.prefix_end,
            sampling,
            step,
            debug_topk,
            debug_layer_stats,
            start_pos,
            &mut scratch,
        );
        let Some(next) = next else {
            break;
        };
        out_tokens.push(next);
        if let Some(cb) = on_token.as_mut() {
            cb(next);
        }
        if should_stop_token(next, &model.config)
            || out_tokens.len() + start_pos >= full.max_seq_len
        {
            break;
        }
    }
    out_tokens
}

fn prefill_full_prompt(
    full: &FullWeights,
    tokens: &[i32],
    cache: &mut [KVCache],
    hidden: &mut [f32],
    prefix_end: Option<usize>,
    start_pos: usize,
) {
    let emb_scale = embedding_scaling(full.model_dim);
    for (offset, &token) in tokens.iter().enumerate() {
        let pos = start_pos + offset;
        let token_id = token as usize;
        if token_id >= full.embedding.rows() {
            continue;
        }
        read_row_f32(&full.embedding, token_id, hidden);
        scale_inplace(hidden, emb_scale);
        let effective_prefix_end = effective_prefix_end_for_position(prefix_end, pos);
        let debug_ctx = if std::env::var("GEMMA_DEBUG_LAYER_STATS")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
            && offset + 1 == tokens.len()
        {
            Some(LayerDebugCtx {
                phase: "prefill_last",
                pos,
                step: 0,
            })
        } else {
            None
        };
        for (layer_idx, layer) in full.layers.iter().enumerate() {
            let next = run_layer(
                full,
                layer,
                layer_idx,
                &mut cache[layer_idx],
                hidden,
                pos,
                effective_prefix_end,
                debug_ctx,
            );
            hidden.copy_from_slice(&next);
        }
    }
}

fn decode_full_next_token(
    model: &Gemma,
    full: &FullWeights,
    cache: &mut [KVCache],
    hidden: &mut [f32],
    out_tokens: &[i32],
    prefix_end: Option<usize>,
    sampling: &mut SamplingState,
    step: usize,
    debug_topk: usize,
    debug_layer_stats: bool,
    start_pos: usize,
    scratch: &mut DecodeScratch,
) -> Option<i32> {
    read_row_f32(&full.final_norm, 0, &mut scratch.norm_scale);
    gemma_ops::nn::rms_norm(hidden, &scratch.norm_scale, 1e-6);

    scratch.logits.resize(full.embedding.rows(), 0.0);
    scratch.tmp_row.resize(full.model_dim, 0.0);
    for row in 0..full.embedding.rows() {
        read_row_f32(&full.embedding, row, &mut scratch.tmp_row);
        let mut sum = 0.0f32;
        for i in 0..full.model_dim {
            sum += hidden[i] * scratch.tmp_row[i];
        }
        scratch.logits[row] = soft_cap(sum, model.config.final_cap);
    }

    if debug_topk > 0 && step == 0 {
        debug_print_topk_logits(model, &scratch.logits, debug_topk);
    }

    let next = sample_token(&mut scratch.logits, sampling) as i32;
    let next_id = next as usize;
    if next_id >= full.embedding.rows() {
        return Some(next);
    }

    if out_tokens.len() + 1 + start_pos >= full.max_seq_len {
        return Some(next);
    }

    read_row_f32(&full.embedding, next_id, hidden);
    scale_inplace(hidden, embedding_scaling(full.model_dim));
    let pos = start_pos + out_tokens.len();
    let effective_prefix_end = effective_prefix_end_for_position(prefix_end, pos);
    for (layer_idx, layer) in full.layers.iter().enumerate() {
        let debug_ctx = if debug_layer_stats && step == 0 {
            Some(LayerDebugCtx {
                phase: "decode_step0",
                pos,
                step,
            })
        } else {
            None
        };
        let next_hidden = run_layer(
            full,
            layer,
            layer_idx,
            &mut cache[layer_idx],
            hidden,
            pos,
            effective_prefix_end,
            debug_ctx,
        );
        hidden.copy_from_slice(&next_hidden);
    }

    Some(next)
}

fn debug_print_topk_logits(model: &Gemma, logits: &[f32], k: usize) {
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(k.min(ranked.len()));
    eprintln!("gemma-debug: first-step top-{k} logits");
    for (rank, (id, logit)) in ranked.iter().enumerate() {
        let piece = model.tokenizer.decode(&[*id as i32]).replace('\n', "\\n");
        eprintln!(
            "  {}: id={} logit={:.6} piece={:?}",
            rank + 1,
            id,
            logit,
            piece
        );
    }
}

fn run_layer(
    full: &FullWeights,
    layer: &LayerWeights,
    layer_idx: usize,
    cache: &mut KVCache,
    input: &[f32],
    pos: usize,
    prefix_end: Option<usize>,
    debug_ctx: Option<LayerDebugCtx>,
) -> Vec<f32> {
    let mut x = input.to_vec();
    maybe_debug_tensor(debug_ctx, layer_idx, "input", input);
    let mut scale = vec![0.0f32; full.model_dim];
    read_row_f32(&layer.pre_att_ns, 0, &mut scale);
    gemma_ops::nn::rms_norm(&mut x, &scale, 1e-6);
    maybe_debug_tensor(debug_ctx, layer_idx, "post_pre_att_norm", &x);

    let mut qkv = vec![0.0f32; (full.heads + 2 * full.kv_heads) * full.qkv_dim];
    matvec(&layer.qkv_ein, &x, &mut qkv);
    let q_len = full.heads * full.qkv_dim;
    let kv_len = full.kv_heads * full.qkv_dim;
    let (q, rest) = qkv.split_at(q_len);
    let (k, v) = rest.split_at(kv_len);

    let mut att_out = vec![0.0f32; full.model_dim];
    let layer_cfg = &full.layer_configs[layer_idx];
    let query_scale = match full.query_scale {
        QueryScaleType::SqrtModelDimDivNumHeads => {
            1.0 / ((full.model_dim as f32) / (layer_cfg.heads.max(1) as f32)).sqrt()
        }
        QueryScaleType::SqrtKeySize => 1.0 / (layer_cfg.qkv_dim.max(1) as f32).sqrt(),
    };
    let layer_window = full
        .attention_window_sizes
        .get(layer_idx)
        .copied()
        .unwrap_or(usize::MAX);
    let plan = build_attention_plan(pos, layer_window, prefix_end, cache.seq_len());
    let att_cap = if full.att_cap > 0.0 {
        Some(full.att_cap)
    } else {
        None
    };
    for h in 0..full.heads {
        let kv_head = kv_head_for_query_head(h, full.heads, full.kv_heads);
        let q_start = h * full.qkv_dim;
        let mut q_vec = q[q_start..q_start + full.qkv_dim].to_vec();
        let k_start = kv_head * full.qkv_dim;
        let mut k_vec = k[k_start..k_start + full.qkv_dim].to_vec();
        if let Some((query_norm, _owner)) = &layer.query_norm_scale {
            if query_norm.cols() == full.qkv_dim {
                let mut q_scale = vec![0.0f32; full.qkv_dim];
                read_row_f32(query_norm, 0, &mut q_scale);
                gemma_ops::nn::rms_norm(&mut q_vec, &q_scale, 1e-6);
            }
        }
        if let Some((key_norm, _owner)) = &layer.key_norm_scale {
            if key_norm.cols() == full.qkv_dim {
                let mut k_scale = vec![0.0f32; full.qkv_dim];
                read_row_f32(key_norm, 0, &mut k_scale);
                gemma_ops::nn::rms_norm(&mut k_vec, &k_scale, 1e-6);
            }
        }
        let v_vec = &v[k_start..k_start + full.qkv_dim];
        let mut ctx = vec![0.0f32; full.qkv_dim];
        run_head_flash_attention(
            &mut q_vec,
            &mut k_vec,
            v_vec,
            cache,
            HeadFlashParams {
                pos,
                kv_head,
                start_pos: plan.start_pos,
                last_pos: plan.last_pos,
                query_scale,
                logits_soft_cap: att_cap,
            },
            &mut ctx,
        );

        let mut att_head = vec![0.0f32; full.model_dim];
        matvec_head(
            &layer.att_ein,
            h,
            full.model_dim,
            full.qkv_dim,
            &ctx,
            &mut att_head,
        );
        for i in 0..full.model_dim {
            att_out[i] += att_head[i];
        }
    }

    if layer_cfg.post_norm == PostNormType::Scale && scale.len() == att_out.len() {
        read_row_f32(&layer.post_att_ns, 0, &mut scale);
        gemma_ops::nn::rms_norm(&mut att_out, &scale, 1e-6);
    }
    maybe_debug_tensor(debug_ctx, layer_idx, "post_post_att_norm", &att_out);
    let mut x_res = input.to_vec();
    for i in 0..full.model_dim {
        x_res[i] += att_out[i];
    }
    maybe_debug_tensor(debug_ctx, layer_idx, "post_att_residual", &x_res);

    let mut ff_in = x_res.clone();
    read_row_f32(&layer.pre_ff_ns, 0, &mut scale);
    gemma_ops::nn::rms_norm(&mut ff_in, &scale, 1e-6);
    maybe_debug_tensor(debug_ctx, layer_idx, "post_pre_ff_norm", &ff_in);

    let mut gating = vec![0.0f32; 2 * full.ff_hidden_dim];
    matvec(&layer.gating_ein, &ff_in, &mut gating);
    let (gate, up) = gating.split_at(full.ff_hidden_dim);

    let mut gated = vec![0.0f32; full.ff_hidden_dim];
    for i in 0..full.ff_hidden_dim {
        let g = gelu(gate[i]);
        gated[i] = g * up[i];
    }

    let mut ff_out = vec![0.0f32; full.model_dim];
    matvec(&layer.linear_w, &gated, &mut ff_out);

    read_row_f32(&layer.post_ff_ns, 0, &mut scale);
    if layer_cfg.post_norm == PostNormType::Scale && scale.len() == ff_out.len() {
        gemma_ops::nn::rms_norm(&mut ff_out, &scale, 1e-6);
    }
    maybe_debug_tensor(debug_ctx, layer_idx, "post_post_ff_norm", &ff_out);

    let mut out = x_res;
    for i in 0..full.model_dim {
        out[i] += ff_out[i];
    }
    maybe_debug_tensor(debug_ctx, layer_idx, "post_ff_residual", &out);

    out
}

fn maybe_debug_tensor(debug_ctx: Option<LayerDebugCtx>, layer_idx: usize, stage: &str, x: &[f32]) {
    let Some(ctx) = debug_ctx else {
        return;
    };
    let (mean, stddev, l2, max_abs) = tensor_stats(x);
    eprintln!(
        "gemma-debug-layer: phase={} step={} pos={} layer={} stage={} mean={:.6} std={:.6} l2={:.6} max_abs={:.6}",
        ctx.phase, ctx.step, ctx.pos, layer_idx, stage, mean, stddev, l2, max_abs
    );
}

fn tensor_stats(x: &[f32]) -> (f32, f32, f32, f32) {
    if x.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n = x.len() as f32;
    let mut sum = 0.0f32;
    let mut sumsq = 0.0f32;
    let mut max_abs = 0.0f32;
    for &v in x {
        sum += v;
        sumsq += v * v;
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    let mean = sum / n;
    let var = (sumsq / n) - mean * mean;
    let stddev = var.max(0.0).sqrt();
    let l2 = sumsq.sqrt();
    (mean, stddev, l2, max_abs)
}

pub(crate) fn matvec(mat: &MatPtr, input: &[f32], out: &mut [f32]) {
    assert_eq!(mat.cols(), input.len());
    assert_eq!(mat.rows(), out.len());
    let bytes = mat.packed_bytes();
    let packed = unsafe { std::slice::from_raw_parts(mat.row_bytes(0), bytes) };
    gemma_ops::matmul::matvec_dispatch(mat.ty(), packed, mat.rows(), mat.cols(), input, out);
}

pub(crate) fn matvec_head(
    mat: &MatPtr,
    head: usize,
    model_dim: usize,
    qkv_dim: usize,
    input: &[f32],
    out: &mut [f32],
) {
    let start_row = head * model_dim;
    let bytes = mat.packed_bytes();
    let packed = unsafe { std::slice::from_raw_parts(mat.row_bytes(0), bytes) };
    let mut row_buf = vec![0.0f32; qkv_dim];
    for r in 0..model_dim {
        let row_idx = start_row + r;
        match mat.ty() {
            Type::F32 => {
                let s = cast_slice::<f32>(packed);
                row_buf.copy_from_slice(&s[row_idx * qkv_dim..(row_idx + 1) * qkv_dim]);
            }
            Type::BF16 => {
                let s = cast_slice::<bf16>(packed);
                let row = &s[row_idx * qkv_dim..(row_idx + 1) * qkv_dim];
                for i in 0..qkv_dim {
                    row_buf[i] = f32::from(row[i]);
                }
            }
            Type::SFP => {
                let start = row_idx * qkv_dim;
                let end = start + qkv_dim;
                sfp::decode_f32(&packed[start..end], &mut row_buf);
            }
            Type::NUQ => {
                let nuq_packed = cast_slice::<NuqStream>(packed);
                nuq::decompress_and_zero_pad_f32(
                    nuq_packed,
                    row_idx * qkv_dim,
                    &mut row_buf,
                    qkv_dim,
                );
            }
            Type::I8 => {
                let i8_packed = cast_slice::<I8Stream>(packed);
                int_format::decompress_and_zero_pad_f32(
                    i8_packed,
                    row_idx * qkv_dim,
                    &mut row_buf,
                    qkv_dim,
                );
            }
            _ => row_buf.fill(0.0),
        }
        let mut sum = 0.0f32;
        for i in 0..qkv_dim {
            sum += row_buf[i] * input[i];
        }
        out[r] = sum;
    }
}

pub(crate) fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
}

pub(crate) fn embedding_scaling(model_dim: usize) -> f32 {
    bf16::from_f32((model_dim as f32).sqrt()).to_f32()
}

pub(crate) fn scale_inplace(values: &mut [f32], scale: f32) {
    if scale == 1.0 {
        return;
    }
    for v in values {
        *v *= scale;
    }
}

pub(crate) fn strip_terminal_turn_markers(mut text: String) -> String {
    for marker in ["<end_of_turn>\n", "<end_of_turn>"] {
        if text.ends_with(marker) {
            text.truncate(text.len() - marker.len());
            break;
        }
    }
    text.trim_end().to_string()
}

pub(crate) fn cast_slice<T: Copy>(bytes: &[u8]) -> &[T] {
    assert_eq!(bytes.len() % core::mem::size_of::<T>(), 0);
    unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr() as *const T,
            bytes.len() / core::mem::size_of::<T>(),
        )
    }
}

pub(crate) fn should_stop_token(token: i32, config: &ModelConfig) -> bool {
    token == config.eos_id || token == config.secondary_eos_id
}

pub(crate) fn soft_cap(x: f32, cap: f32) -> f32 {
    if cap > 0.0 {
        cap * (x / cap).tanh()
    } else {
        x
    }
}

#[cfg(test)]
mod cache_state_tests {
    use super::*;
    use gemma_util::basics::Extents2D;

    fn dummy_mat(name: &str) -> MatPtr {
        MatPtr::new(name, Type::F32, Extents2D::new(0, 0))
    }

    fn dummy_layer() -> LayerWeights {
        LayerWeights {
            qkv_ein: dummy_mat("qkv"),
            qkv_owner: MatOwner::new(),
            att_ein: dummy_mat("att"),
            att_owner: MatOwner::new(),
            gating_ein: dummy_mat("gating"),
            gating_owner: MatOwner::new(),
            linear_w: dummy_mat("linear"),
            linear_owner: MatOwner::new(),
            pre_att_ns: dummy_mat("pre_att_ns"),
            pre_att_owner: MatOwner::new(),
            pre_ff_ns: dummy_mat("pre_ff_ns"),
            pre_ff_owner: MatOwner::new(),
            post_att_ns: dummy_mat("post_att_ns"),
            post_att_owner: MatOwner::new(),
            post_ff_ns: dummy_mat("post_ff_ns"),
            post_ff_owner: MatOwner::new(),
            key_norm_scale: None,
            query_norm_scale: None,
        }
    }

    fn dummy_full(layer_count: usize, max_seq_len: usize) -> FullWeights {
        let layers = (0..layer_count).map(|_| dummy_layer()).collect();
        FullWeights {
            embedding: dummy_mat("embed"),
            _embedding_owner: MatOwner::new(),
            final_norm: dummy_mat("final_norm"),
            _final_norm_owner: MatOwner::new(),
            layers,
            model_dim: 1,
            heads: 1,
            kv_heads: 1,
            qkv_dim: 1,
            ff_hidden_dim: 1,
            max_seq_len,
            attention_window_sizes: vec![max_seq_len; layer_count],
            att_cap: 0.0,
            query_scale: QueryScaleType::SqrtKeySize,
            layer_configs: (0..layer_count)
                .map(|idx| crate::configs::LayerConfig {
                    layer_idx: idx,
                    attention_type: crate::configs::LayerAttentionType::Gemma,
                    post_norm: crate::configs::PostNormType::None,
                    use_qk_norm: false,
                    ff_biases: false,
                    model_dim: 1,
                    ff_hidden_dim: 1,
                    heads: 1,
                    kv_heads: 1,
                    qkv_dim: 2,
                    optimized_gating: true,
                    activation: crate::configs::ActivationType::Gelu,
                    post_qk: crate::configs::PostQKType::Rope,
                })
                .collect(),
        }
    }

    #[test]
    fn cache_state_allocates_per_layer() {
        let full = dummy_full(3, 8);
        let state = CacheState::new(&full);
        assert_eq!(state.cache.len(), 3);
        for kv in state.cache.iter() {
            assert_eq!(kv.seq_len(), 8);
        }
        assert_eq!(state.pos, 0);
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use gemma_util::basics::Extents2D;
    use kitoken::{
        Configuration, Fallback, Kitoken, Metadata, Model, SpecialToken, SpecialTokenKind,
        SpecialVocab, Token, Vocab,
    };

    fn owned_mat(name: &str, rows: usize, cols: usize, fill: f32) -> (MatPtr, Vec<f32>) {
        let mut mat = MatPtr::new(name, Type::F32, Extents2D::new(rows, cols));
        let mut buf = vec![fill; rows * cols];
        mat.set_ptr(buf.as_mut_ptr() as *mut u8, cols);
        (mat, buf)
    }

    fn make_full_for_test() -> (FullWeights, Vec<Vec<f32>>) {
        let mut bufs: Vec<Vec<f32>> = Vec::new();
        let (embedding, b0) = owned_mat("embed", 4, 1, 0.0);
        bufs.push(b0);
        let (final_norm, b1) = owned_mat("final_norm", 1, 1, 1.0);
        bufs.push(b1);

        let (qkv_ein, b2) = owned_mat("qkv", 6, 1, 0.0);
        let (att_ein, b3) = owned_mat("att", 1, 2, 0.0);
        let (gating_ein, b4) = owned_mat("gating", 2, 1, 0.0);
        let (linear_w, b5) = owned_mat("linear", 1, 1, 0.0);
        let (pre_att_ns, b6) = owned_mat("pre_att_ns", 1, 1, 1.0);
        let (pre_ff_ns, b7) = owned_mat("pre_ff_ns", 1, 1, 1.0);
        let (post_att_ns, b8) = owned_mat("post_att_ns", 1, 1, 1.0);
        let (post_ff_ns, b9) = owned_mat("post_ff_ns", 1, 1, 1.0);
        bufs.extend([b2, b3, b4, b5, b6, b7, b8, b9]);

        let layer = LayerWeights {
            qkv_ein,
            qkv_owner: MatOwner::new(),
            att_ein,
            att_owner: MatOwner::new(),
            gating_ein,
            gating_owner: MatOwner::new(),
            linear_w,
            linear_owner: MatOwner::new(),
            pre_att_ns,
            pre_att_owner: MatOwner::new(),
            pre_ff_ns,
            pre_ff_owner: MatOwner::new(),
            post_att_ns,
            post_att_owner: MatOwner::new(),
            post_ff_ns,
            post_ff_owner: MatOwner::new(),
            key_norm_scale: None,
            query_norm_scale: None,
        };

        let full = FullWeights {
            embedding,
            _embedding_owner: MatOwner::new(),
            final_norm,
            _final_norm_owner: MatOwner::new(),
            layers: vec![layer],
            model_dim: 1,
            heads: 1,
            kv_heads: 1,
            qkv_dim: 2,
            ff_hidden_dim: 1,
            max_seq_len: 8,
            attention_window_sizes: vec![8],
            att_cap: 0.0,
            query_scale: QueryScaleType::SqrtKeySize,
            layer_configs: vec![crate::configs::LayerConfig {
                layer_idx: 0,
                attention_type: crate::configs::LayerAttentionType::Gemma,
                post_norm: crate::configs::PostNormType::None,
                use_qk_norm: false,
                ff_biases: false,
                model_dim: 1,
                ff_hidden_dim: 1,
                heads: 1,
                kv_heads: 1,
                qkv_dim: 1,
                optimized_gating: true,
                activation: crate::configs::ActivationType::Gelu,
                post_qk: crate::configs::PostQKType::HalfRope,
            }],
        };

        (full, bufs)
    }

    fn mock_tokenizer() -> Tokenizer {
        let vocab: Vocab = vec![
            Token {
                id: 0,
                bytes: b"a".to_vec(),
            },
            Token {
                id: 1,
                bytes: b"b".to_vec(),
            },
        ];
        let specials: SpecialVocab = vec![
            SpecialToken {
                id: 2,
                bytes: b"<s>".to_vec(),
                kind: SpecialTokenKind::Control,
                ident: None,
                score: 0.0,
                extract: false,
            },
            SpecialToken {
                id: 3,
                bytes: b"</s>".to_vec(),
                kind: SpecialTokenKind::Control,
                ident: None,
                score: 0.0,
                extract: false,
            },
        ];
        let config = Configuration {
            fallback: vec![Fallback::Unknown],
            normalization: Vec::new(),
            split: Vec::new(),
            processing: Vec::new(),
            decoding: Vec::new(),
            templates: Vec::new(),
        };
        let meta = Metadata::default();
        let model = Model::WordPiece {
            vocab,
            max_word_chars: 16,
        };
        let kt = Kitoken::new(model, specials, config, meta).expect("mock kitoken");
        Tokenizer::from_kitoken(kt)
    }

    #[test]
    fn streaming_callback_receives_token() {
        let tokenizer = mock_tokenizer();
        let (full, _bufs) = make_full_for_test();
        let mut config = ModelConfig::new(1);
        config.model_dim = 1;
        config.vocab_size = 4;
        config.max_seq_len = 8;
        config.eos_id = 3;
        config.secondary_eos_id = 3;
        config.att_cap = 0.0;
        config.final_cap = 0.0;
        config.query_scale = QueryScaleType::SqrtKeySize;
        config.attention_window_sizes = vec![8];
        config.layers = full.layer_configs.clone();

        let model = Gemma {
            config,
            tokenizer,
            weights: None,
            full: Some(full),
        };
        let mut state = model.init_cache_state().unwrap();

        // BOS (2) + token "a" (0)
        let chunk_tokens = vec![2, 0];
        let mut seen: Vec<i32> = Vec::new();
        let mut cb = |tok: i32| seen.push(tok);
        let mut cb_ref: &mut dyn FnMut(i32) = &mut cb;
        let _ = generate_full_tokens_with_cache(
            &model,
            model.full.as_ref().unwrap(),
            &chunk_tokens,
            1,
            GenerationOptions {
                prefix_end: None,
                sampling: SamplingOptions {
                    temperature: 1.0,
                    top_k: 1,
                    top_p: 1.0,
                    seed: Some(123),
                },
            },
            &mut SamplingState::new(SamplingOptions::default()),
            state.pos,
            Some(&mut state.cache),
            Some(&mut cb_ref),
        );
        assert!(!seen.is_empty());
    }
}

fn resolve_prefix_end(prefix_end: Option<usize>, available_tokens: usize) -> Option<usize> {
    prefix_end.map(|p| p.min(available_tokens))
}

pub(crate) fn effective_prefix_end_for_position(
    prefix_end: Option<usize>,
    pos: usize,
) -> Option<usize> {
    resolve_prefix_end(prefix_end, pos.saturating_add(1))
}

pub(crate) fn sample_token(logits: &mut [f32], sampling: &mut SamplingState) -> usize {
    if logits.is_empty() {
        return 0;
    }
    let opts = sampling.opts;
    if opts.top_k <= 1 || opts.temperature <= 0.0 {
        return argmax_index(logits);
    }

    let mut candidates: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v / opts.temperature))
        .collect();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    let k = opts.top_k.min(candidates.len());
    candidates.truncate(k);

    if opts.top_p < 1.0 {
        let max_logit = candidates[0].1;
        let mut probs: Vec<f32> = candidates
            .iter()
            .map(|(_, l)| (*l - max_logit).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
            let mut cum = 0.0f32;
            let mut keep = probs.len();
            for (i, p) in probs.iter().enumerate() {
                cum += *p;
                if cum >= opts.top_p {
                    keep = i + 1;
                    break;
                }
            }
            candidates.truncate(keep.max(1));
        }
    }

    let max_logit = candidates[0].1;
    let mut weights = vec![0.0f32; candidates.len()];
    let mut sum = 0.0f32;
    for (i, (_, l)) in candidates.iter().enumerate() {
        let w = (*l - max_logit).exp();
        weights[i] = w;
        sum += w;
    }
    if sum <= 0.0 {
        return candidates[0].0;
    }
    let mut r = sampling.rng.next_f32() * sum;
    for (i, w) in weights.iter().enumerate() {
        r -= *w;
        if r <= 0.0 {
            return candidates[i].0;
        }
    }
    candidates[candidates.len() - 1].0
}

fn argmax_index(values: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::{
        argmax_index, effective_prefix_end_for_position, resolve_prefix_end, sample_token,
        should_stop_token, soft_cap, SamplingOptions, SamplingState,
    };
    use crate::configs::ModelConfig;

    #[test]
    fn eos_stop_token_logic() {
        let mut cfg = ModelConfig::new(0);
        cfg.eos_id = 1;
        cfg.secondary_eos_id = 2;
        assert!(should_stop_token(1, &cfg));
        assert!(should_stop_token(2, &cfg));
        assert!(!should_stop_token(3, &cfg));
    }

    #[test]
    fn soft_cap_behaves_as_expected() {
        assert_eq!(soft_cap(3.0, 0.0), 3.0);
        let capped = soft_cap(100.0, 5.0);
        assert!(capped <= 5.0);
        assert!(capped > 0.0);
    }

    #[test]
    fn sampling_with_seed_is_deterministic() {
        let opts = SamplingOptions {
            temperature: 0.8,
            top_k: 3,
            top_p: 0.95,
            seed: Some(1234),
        };
        let mut a = SamplingState::new(opts);
        let mut b = SamplingState::new(opts);
        let mut logits_a = vec![1.0, 2.0, 0.5, -1.0];
        let mut logits_b = logits_a.clone();
        let mut seq_a = Vec::new();
        let mut seq_b = Vec::new();
        for _ in 0..10 {
            seq_a.push(sample_token(&mut logits_a, &mut a));
            seq_b.push(sample_token(&mut logits_b, &mut b));
        }
        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn argmax_when_topk_one() {
        assert_eq!(argmax_index(&[0.1, 2.0, 1.0]), 1);
        let opts = SamplingOptions {
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            seed: Some(7),
        };
        let mut state = SamplingState::new(opts);
        let mut logits = vec![0.1, 2.0, 1.0];
        assert_eq!(sample_token(&mut logits, &mut state), 1);
    }

    #[test]
    fn resolve_prefix_end_clamps_to_available_tokens() {
        assert_eq!(resolve_prefix_end(None, 4), None);
        assert_eq!(resolve_prefix_end(Some(2), 4), Some(2));
        assert_eq!(resolve_prefix_end(Some(8), 4), Some(4));
    }

    #[test]
    fn prefix_end_position_semantics_match_prefill_and_decode_rules() {
        // Prefill position 0 has one available token.
        assert_eq!(
            effective_prefix_end_for_position(Some(usize::MAX), 0),
            Some(1)
        );
        // Decode position N has N+1 total available tokens after appending next token.
        assert_eq!(
            effective_prefix_end_for_position(Some(usize::MAX), 5),
            Some(6)
        );
        assert_eq!(effective_prefix_end_for_position(Some(3), 5), Some(3));
        assert_eq!(effective_prefix_end_for_position(None, 5), None);
    }
}
