//! GPU-accelerated inference using the `Backend` trait from `gemma-gpu`.
//!
//! This module provides a GPU-accelerated generation path that offloads
//! the three heavy matrix-vector multiplies per layer (qkv_ein, gating_ein,
//! linear_w) to a GPU backend while keeping attention (RoPE, KV cache,
//! flash attention) on the CPU.
//!
//! # Memory management
//!
//! Weight matrices are uploaded to the GPU at model load time. If the device
//! runs out of memory, remaining layers fall back to CPU matvec. A status
//! message reports how many layers are GPU-resident.

use gemma_gpu::backend::{Backend, Result as GpuResult};

use crate::attention::{build_attention_plan, kv_head_for_query_head};
use crate::configs::{PostNormType, QueryScaleType};
use crate::flash_attention::{run_head_flash_attention, HeadFlashParams};
use crate::gemma::{
    effective_prefix_end_for_position, embedding_scaling, matvec_head, read_row_f32, sample_token,
    scale_inplace, should_stop_token, soft_cap, FullWeights, LayerWeights, SamplingState,
};
use crate::gemma::{strip_terminal_turn_markers, CacheState, Gemma, SamplingOptions};
use crate::kv_cache::KVCache;

/// GPU-resident weight matrices for a single transformer layer.
pub struct GpuLayerWeights<B: Backend> {
    pub qkv_ein: B::Wgt,
    pub gating_ein: B::Wgt,
    pub linear_w: B::Wgt,
}

/// Pre-allocated GPU scratch buffers for matvec input/output.
///
/// These are reused across layers and tokens to avoid repeated allocation.
pub struct GpuBuffers<B: Backend> {
    /// Input buffer for qkv_ein and gating_ein (model_dim elements).
    pub model_dim_buf: B::Buf,
    /// Output buffer for qkv_ein (qkv_total elements).
    pub qkv_buf: B::Buf,
    /// Output buffer for gating_ein (2 * ff_hidden_dim elements).
    pub gating_buf: B::Buf,
    /// Input buffer for linear_w (ff_hidden_dim elements).
    pub ff_hidden_buf: B::Buf,
    /// Output buffer for linear_w (model_dim elements).
    pub linear_out_buf: B::Buf,
    /// Logits buffer (vocab elements) when sampling on GPU.
    pub logits_buf: Option<B::Buf>,
}

/// GPU-accelerated model holding uploaded weights and scratch buffers.
pub struct GpuModel<B: Backend> {
    pub backend: B,
    /// Per-layer GPU weights. `None` for layers that didn't fit in VRAM.
    pub layers: Vec<Option<GpuLayerWeights<B>>>,
    /// Pre-allocated scratch buffers for GPU matvec operations.
    pub bufs: GpuBuffers<B>,
    /// Optional GPU-resident embedding matrix for logits matvec.
    pub embedding_wgt: Option<B::Wgt>,
    /// How many layers have GPU-resident weights.
    gpu_layer_count: usize,
    /// Whether to attempt GPU logits (softmax still on host).
    gpu_logits_enabled: bool,
}

struct GpuDecodeScratch {
    norm_scale: Vec<f32>,
    logits: Vec<f32>,
    tmp_row: Vec<f32>,
}

impl GpuDecodeScratch {
    fn new(full: &FullWeights) -> Self {
        Self {
            norm_scale: vec![0.0; full.model_dim],
            logits: vec![0.0; full.embedding.rows()],
            tmp_row: vec![0.0; full.model_dim],
        }
    }
}

fn compute_logits_cpu(
    full: &FullWeights,
    model: &Gemma,
    hidden: &[f32],
    logits_out: &mut Vec<f32>,
) {
    logits_out.resize(full.embedding.rows(), 0.0);
    let mut tmp_row = vec![0.0f32; full.model_dim];
    for row in 0..full.embedding.rows() {
        read_row_f32(&full.embedding, row, &mut tmp_row);
        let mut sum = 0.0f32;
        for i in 0..full.model_dim {
            sum += hidden[i] * tmp_row[i];
        }
        logits_out[row] = soft_cap(sum, model.config.final_cap);
    }
}

impl<B: Backend> GpuModel<B> {
    /// Upload model weights to the GPU backend.
    ///
    /// Attempts to upload all layers. If the device runs out of memory,
    /// remaining layers will use CPU fallback during inference.
    pub(crate) fn new(backend: B, full: &FullWeights) -> GpuResult<Self> {
        let qkv_total = (full.heads + 2 * full.kv_heads) * full.qkv_dim;

        // Allocate scratch buffers
        let bufs = GpuBuffers {
            model_dim_buf: backend.alloc(full.model_dim)?,
            qkv_buf: backend.alloc(qkv_total)?,
            gating_buf: backend.alloc(2 * full.ff_hidden_dim)?,
            ff_hidden_buf: backend.alloc(full.ff_hidden_dim)?,
            linear_out_buf: backend.alloc(full.model_dim)?,
            logits_buf: backend.alloc(full.embedding.rows()).ok(),
        };

        let mut layers = Vec::with_capacity(full.layers.len());
        let mut gpu_layer_count = 0;

        for (idx, layer) in full.layers.iter().enumerate() {
            match upload_layer_weights(&backend, layer) {
                Ok(gpu_layer) => {
                    layers.push(Some(gpu_layer));
                    gpu_layer_count += 1;
                }
                Err(e) => {
                    eprintln!(
                        "GPU: layer {} upload failed ({e}), remaining layers on CPU",
                        idx
                    );
                    layers.push(None);
                    for _ in (idx + 1)..full.layers.len() {
                        layers.push(None);
                    }
                    break;
                }
            }
        }

        let caps = backend.caps();
        eprintln!(
            "GPU: {} layers on {} ({}/{}), {} layers on CPU",
            gpu_layer_count,
            caps.name,
            gpu_layer_count,
            full.layers.len(),
            full.layers.len() - gpu_layer_count,
        );

        // Upload embedding for optional GPU logits matvec.
        let embedding_wgt = upload_mat(&backend, &full.embedding).ok();
        let gpu_logits_enabled = std::env::var("GEMMA_GPU_CPU_LOGITS")
            .ok()
            .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .unwrap_or(true);

        Ok(Self {
            backend,
            layers,
            bufs,
            embedding_wgt,
            gpu_layer_count,
            gpu_logits_enabled,
        })
    }

    /// Number of layers with GPU-resident weights.
    pub fn gpu_layer_count(&self) -> usize {
        self.gpu_layer_count
    }
}

/// Run a GPU matvec: upload input → compute → sync → download output.
///
/// Free function to avoid borrow conflicts between GpuModel fields.
fn gpu_matvec<B: Backend>(
    backend: &B,
    wgt: &B::Wgt,
    input: &[f32],
    output: &mut [f32],
    in_buf: &mut B::Buf,
    out_buf: &mut B::Buf,
) -> GpuResult<()> {
    backend.upload_f32(input, in_buf)?;
    backend.matvec(wgt, in_buf, out_buf)?;
    backend.synchronize()?;
    backend.download_f32(out_buf, output)
}

/// Upload a single layer's weight matrices to the GPU.
fn upload_layer_weights<B: Backend>(
    backend: &B,
    layer: &LayerWeights,
) -> GpuResult<GpuLayerWeights<B>> {
    let qkv_ein = upload_mat(backend, &layer.qkv_ein)?;
    let gating_ein = upload_mat(backend, &layer.gating_ein)?;
    let linear_w = upload_mat(backend, &layer.linear_w)?;
    Ok(GpuLayerWeights {
        qkv_ein,
        gating_ein,
        linear_w,
    })
}

/// Upload a MatPtr weight matrix to the GPU backend.
fn upload_mat<B: Backend>(backend: &B, mat: &gemma_util::mat::MatPtr) -> GpuResult<B::Wgt> {
    let bytes = mat.packed_bytes();
    let packed = unsafe { std::slice::from_raw_parts(mat.row_bytes(0), bytes) };
    backend.upload_weight(packed, mat.ty(), mat.rows(), mat.cols())
}

// ── Generation ──────────────────────────────────────────────────────────

impl Gemma {
    /// Create a GPU model by uploading weights to the given backend.
    ///
    /// Returns `None` if full weights are not loaded.
    pub fn create_gpu_model<B: Backend>(&self, backend: B) -> Option<GpuResult<GpuModel<B>>> {
        self.full.as_ref().map(|full| GpuModel::new(backend, full))
    }

    /// Generate text using GPU-accelerated inference.
    ///
    /// The heavy matvecs (qkv, gating, linear) run on the GPU backend,
    /// while attention and sampling remain on CPU.
    pub fn generate_with_gpu<B: Backend>(
        &self,
        gpu: &mut GpuModel<B>,
        prompt: &str,
        max_tokens: usize,
        sampling: SamplingOptions,
    ) -> String {
        let full = match &self.full {
            Some(f) => f,
            None => return format!("{prompt} [no full weights]"),
        };

        let tokens = self.wrap_tokens(self.tokenizer.encode(prompt), 0);
        if tokens.is_empty() || max_tokens == 0 {
            return prompt.to_string();
        }

        let mut samp = SamplingState::new(sampling);
        let out_tokens = generate_gpu_tokens(self, full, gpu, &tokens, max_tokens, &mut samp);

        let generated = if out_tokens.len() > tokens.len() {
            self.tokenizer.decode(&out_tokens[tokens.len()..])
        } else {
            String::new()
        };
        strip_turn_markers(generated)
    }

    /// Generate from a pre-formatted conversation string using GPU.
    ///
    /// Like `generate_chat` but uses the GPU backend for matvec operations.
    /// The caller builds the conversation with BOS and turn markers.
    pub fn generate_chat_gpu<B: Backend>(
        &self,
        gpu: &mut GpuModel<B>,
        formatted_conversation: &str,
        max_tokens: usize,
        sampling: SamplingOptions,
    ) -> String {
        let full = match &self.full {
            Some(f) => f,
            None => return String::new(),
        };

        let mut tokens = self.tokenizer.encode(formatted_conversation);
        if tokens.is_empty() || max_tokens == 0 {
            return String::new();
        }
        // Prepend BOS token — wrap_tokens normally does this but we bypass it.
        const BOS_ID: i32 = 2;
        tokens.insert(0, BOS_ID);
        let prompt_len = tokens.len();
        let mut samp = SamplingState::new(sampling);
        let out_tokens = generate_gpu_tokens(self, full, gpu, &tokens, max_tokens, &mut samp);

        let generated = if out_tokens.len() > prompt_len {
            self.tokenizer.decode(&out_tokens[prompt_len..])
        } else {
            String::new()
        };
        strip_turn_markers(generated)
    }

    /// Generate a chat response on GPU using an existing KV cache.
    ///
    /// `chunk_tokens` should contain only the newly appended conversation
    /// segment for this turn (including markers). Insert BOS only on the first call.
    pub fn generate_chat_gpu_with_cache<B: Backend>(
        &self,
        gpu: &mut GpuModel<B>,
        state: &mut CacheState,
        chunk_tokens: &[i32],
        max_tokens: usize,
        sampling: SamplingOptions,
    ) -> Option<String> {
        let full = self.full.as_ref()?;
        let mut samp = SamplingState::new(sampling);
        let out_tokens = generate_gpu_tokens_with_cache(
            self,
            full,
            gpu,
            chunk_tokens,
            max_tokens,
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

    /// Streaming variant of GPU chat with cache persistence.
    pub fn generate_chat_gpu_streaming_with_cache<B: Backend>(
        &self,
        gpu: &mut GpuModel<B>,
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
            if crate::gemma::should_stop_token(tok, &self.config) {
                return;
            }
            let piece = self.tokenizer.decode(&[tok]);
            if !piece.is_empty() {
                on_text(&piece);
                assembled.push_str(&piece);
            }
        };
        let mut cb_ref: &mut dyn FnMut(i32) = &mut stream_cb;
        let out_tokens = generate_gpu_tokens_with_cache(
            self,
            full,
            gpu,
            chunk_tokens,
            max_tokens,
            &mut samp,
            state.pos,
            Some(&mut state.cache),
            Some(&mut cb_ref),
        );
        state.pos += out_tokens.len();
        Some(strip_terminal_turn_markers(assembled))
    }
}

fn generate_gpu_tokens<B: Backend>(
    model: &Gemma,
    full: &FullWeights,
    gpu: &mut GpuModel<B>,
    tokens: &[i32],
    max_tokens: usize,
    sampling: &mut SamplingState,
) -> Vec<i32> {
    generate_gpu_tokens_with_cache(
        model, full, gpu, tokens, max_tokens, sampling, 0, None, None,
    )
}

fn generate_gpu_tokens_with_cache<B: Backend>(
    model: &Gemma,
    full: &FullWeights,
    gpu: &mut GpuModel<B>,
    tokens: &[i32],
    max_tokens: usize,
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
    let mut scratch = GpuDecodeScratch::new(full);
    let mut out_tokens = tokens.to_vec();

    // Prefill: process each prompt token through all layers
    let emb_scale = embedding_scaling(full.model_dim);
    for (offset, &token) in tokens.iter().enumerate() {
        let pos = start_pos + offset;
        let token_id = token as usize;
        if token_id >= full.embedding.rows() {
            continue;
        }
        read_row_f32(&full.embedding, token_id, &mut hidden);
        scale_inplace(&mut hidden, emb_scale);
        let prefix_end = effective_prefix_end_for_position(None, pos);
        for (layer_idx, layer) in full.layers.iter().enumerate() {
            let gpu_layer = gpu.layers[layer_idx].as_ref();
            let next = run_layer_gpu(
                full,
                layer,
                layer_idx,
                &mut cache[layer_idx],
                &hidden,
                pos,
                prefix_end,
                gpu_layer,
                &gpu.backend,
                &mut gpu.bufs,
            );
            hidden.copy_from_slice(&next);
        }
    }

    // Decode: generate tokens one at a time
    for _step in 0..max_tokens {
        // Final norm + logits
        read_row_f32(&full.final_norm, 0, &mut scratch.norm_scale);
        gemma_ops::nn::rms_norm(&mut hidden, &scratch.norm_scale, 1e-6);

        let next = if gpu.gpu_logits_enabled
            && gpu.embedding_wgt.is_some()
            && gpu.bufs.logits_buf.is_some()
        {
            let embed_wgt = gpu.embedding_wgt.as_ref().unwrap();
            let logits_buf = gpu.bufs.logits_buf.as_mut().unwrap();
            let mut download_and_sample =
                |reason: &str, logits_buf: &mut <B as Backend>::Buf| -> i32 {
                    if !reason.is_empty() {
                        eprintln!("{reason}");
                    }
                    scratch.logits.resize(full.embedding.rows(), 0.0);
                    if let Err(e) = gpu
                        .backend
                        .download_f32(logits_buf, &mut scratch.logits[..full.embedding.rows()])
                    {
                        eprintln!("GPU logits download failed, falling back to CPU logits: {e}");
                        compute_logits_cpu(full, model, &hidden, &mut scratch.logits);
                    }
                    for v in &mut scratch.logits {
                        *v = soft_cap(*v, model.config.final_cap);
                    }
                    sample_token(&mut scratch.logits, sampling) as i32
                };
            // GPU matvec for logits, then argmax on GPU backend (download just index)
            if let Err(e) = gpu.backend.upload_f32(&hidden, &mut gpu.bufs.model_dim_buf) {
                eprintln!("GPU logits upload failed, falling back to CPU logits: {e}");
                compute_logits_cpu(full, model, &hidden, &mut scratch.logits);
                sample_token(&mut scratch.logits, sampling) as i32
            } else if let Err(e) =
                gpu.backend
                    .matvec(embed_wgt, &gpu.bufs.model_dim_buf, logits_buf)
            {
                eprintln!("GPU logits matvec failed, falling back to CPU logits: {e}");
                compute_logits_cpu(full, model, &hidden, &mut scratch.logits);
                sample_token(&mut scratch.logits, sampling) as i32
            } else if let Err(e) = gpu.backend.synchronize() {
                eprintln!("GPU sync after logits failed, falling back to CPU logits: {e}");
                compute_logits_cpu(full, model, &hidden, &mut scratch.logits);
                sample_token(&mut scratch.logits, sampling) as i32
            } else {
                // argmax is invariant under monotonic soft_cap(), so we can keep logits on device.
                match gpu.backend.argmax_with_scratch(logits_buf, None) {
                    Ok(idx) => idx as i32,
                    Err(e) => download_and_sample(
                        &format!("device argmax failed, falling back to host sample: {e}"),
                        logits_buf,
                    ),
                }
            }
        } else {
            compute_logits_cpu(full, model, &hidden, &mut scratch.logits);
            sample_token(&mut scratch.logits, sampling) as i32
        };
        let next_id = next as usize;
        out_tokens.push(next);
        if let Some(cb) = on_token.as_mut() {
            cb(next);
        }
        if should_stop_token(next, &model.config)
            || out_tokens.len() + start_pos >= full.max_seq_len
        {
            break;
        }
        if next_id >= full.embedding.rows() {
            break;
        }

        // Prepare next token and run through layers
        read_row_f32(&full.embedding, next_id, &mut hidden);
        scale_inplace(&mut hidden, emb_scale);
        let pos = start_pos + out_tokens.len() - 1;
        let prefix_end = effective_prefix_end_for_position(None, pos);
        for (layer_idx, layer) in full.layers.iter().enumerate() {
            let gpu_layer = gpu.layers[layer_idx].as_ref();
            let next_hidden = run_layer_gpu(
                full,
                layer,
                layer_idx,
                &mut cache[layer_idx],
                &hidden,
                pos,
                prefix_end,
                gpu_layer,
                &gpu.backend,
                &mut gpu.bufs,
            );
            hidden.copy_from_slice(&next_hidden);
        }
    }

    out_tokens
}

/// Run a single transformer layer with optional GPU matvec acceleration.
///
/// When `gpu_layer` is Some, the three big matvecs (qkv_ein, gating_ein,
/// linear_w) are executed on the GPU via upload→compute→download.
/// Everything else (norms, attention, GELU, residuals) runs on CPU.
fn run_layer_gpu<B: Backend>(
    full: &FullWeights,
    layer: &LayerWeights,
    layer_idx: usize,
    cache: &mut KVCache,
    input: &[f32],
    pos: usize,
    prefix_end: Option<usize>,
    gpu_layer: Option<&GpuLayerWeights<B>>,
    backend: &B,
    bufs: &mut GpuBuffers<B>,
) -> Vec<f32> {
    let mut x = input.to_vec();

    // ── Pre-attention RMS norm (CPU) ─────────────────────────────────
    let mut scale = vec![0.0f32; full.model_dim];
    read_row_f32(&layer.pre_att_ns, 0, &mut scale);
    gemma_ops::nn::rms_norm(&mut x, &scale, 1e-6);

    // ── QKV projection (GPU or CPU) ─────────────────────────────────
    let qkv_total = (full.heads + 2 * full.kv_heads) * full.qkv_dim;
    let mut qkv = vec![0.0f32; qkv_total];
    if let Some(gl) = gpu_layer {
        gpu_matvec(
            backend,
            &gl.qkv_ein,
            &x,
            &mut qkv,
            &mut bufs.model_dim_buf,
            &mut bufs.qkv_buf,
        )
        .unwrap_or_else(|e| {
            eprintln!("GPU qkv matvec failed, CPU fallback: {e}");
            crate::gemma::matvec(&layer.qkv_ein, &x, &mut qkv);
        });
    } else {
        crate::gemma::matvec(&layer.qkv_ein, &x, &mut qkv);
    }

    // ── Attention (entirely on CPU) ─────────────────────────────────
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

        // Per-head attention output projection (CPU — small matrix)
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

    // ── Post-attention norm + residual (CPU) ────────────────────────
    if layer_cfg.post_norm == PostNormType::Scale && scale.len() == att_out.len() {
        read_row_f32(&layer.post_att_ns, 0, &mut scale);
        gemma_ops::nn::rms_norm(&mut att_out, &scale, 1e-6);
    }
    let mut x_res = input.to_vec();
    for i in 0..full.model_dim {
        x_res[i] += att_out[i];
    }

    // ── Pre-FF RMS norm (CPU) ───────────────────────────────────────
    let mut ff_in = x_res.clone();
    read_row_f32(&layer.pre_ff_ns, 0, &mut scale);
    gemma_ops::nn::rms_norm(&mut ff_in, &scale, 1e-6);

    // ── Gating projection (GPU or CPU) ──────────────────────────────
    let mut gating = vec![0.0f32; 2 * full.ff_hidden_dim];
    if let Some(gl) = gpu_layer {
        gpu_matvec(
            backend,
            &gl.gating_ein,
            &ff_in,
            &mut gating,
            &mut bufs.model_dim_buf,
            &mut bufs.gating_buf,
        )
        .unwrap_or_else(|e| {
            eprintln!("GPU gating matvec failed, CPU fallback: {e}");
            crate::gemma::matvec(&layer.gating_ein, &ff_in, &mut gating);
        });
    } else {
        crate::gemma::matvec(&layer.gating_ein, &ff_in, &mut gating);
    }

    // ── GELU gate (CPU) ─────────────────────────────────────────────
    let (gate, up) = gating.split_at(full.ff_hidden_dim);
    let mut gated = vec![0.0f32; full.ff_hidden_dim];
    for i in 0..full.ff_hidden_dim {
        let g = crate::gemma::gelu(gate[i]);
        gated[i] = g * up[i];
    }

    // ── Linear projection (GPU or CPU) ──────────────────────────────
    let mut ff_out = vec![0.0f32; full.model_dim];
    if let Some(gl) = gpu_layer {
        gpu_matvec(
            backend,
            &gl.linear_w,
            &gated,
            &mut ff_out,
            &mut bufs.ff_hidden_buf,
            &mut bufs.linear_out_buf,
        )
        .unwrap_or_else(|e| {
            eprintln!("GPU linear matvec failed, CPU fallback: {e}");
            crate::gemma::matvec(&layer.linear_w, &gated, &mut ff_out);
        });
    } else {
        crate::gemma::matvec(&layer.linear_w, &gated, &mut ff_out);
    }

    // ── Post-FF norm + residual (CPU) ───────────────────────────────
    if layer_cfg.post_norm == PostNormType::Scale && scale.len() == ff_out.len() {
        read_row_f32(&layer.post_ff_ns, 0, &mut scale);
        gemma_ops::nn::rms_norm(&mut ff_out, &scale, 1e-6);
    }
    let mut out = x_res;
    for i in 0..full.model_dim {
        out[i] += ff_out[i];
    }

    out
}

fn strip_turn_markers(mut text: String) -> String {
    for marker in ["<end_of_turn>\n", "<end_of_turn>"] {
        if text.ends_with(marker) {
            text.truncate(text.len() - marker.len());
            break;
        }
    }
    text.trim_end().to_string()
}
