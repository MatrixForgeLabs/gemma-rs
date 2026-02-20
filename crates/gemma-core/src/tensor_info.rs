//! Tensor metadata and registry (minimal).

use std::collections::HashMap;

use gemma_compression::types::Type;
use gemma_util::basics::Extents2D;

use crate::configs::{LayerConfig, ModelConfig};

#[derive(Clone, Debug, Default)]
pub struct TensorInfo {
    pub base_name: String,
    pub source_names: Vec<String>,
    pub preshape: Vec<usize>,
    pub axes: Vec<usize>,
    pub shape: Vec<usize>,
    pub concat_names: Vec<String>,
    pub concat_axis: usize,
    pub min_size: Type,
    pub scaled_softplus: bool,
    pub cols_take_extra_dims: bool,
}

pub fn extents_from_info(tensor: Option<&TensorInfo>) -> Extents2D {
    let tensor = match tensor {
        Some(t) => t,
        None => return Extents2D::new(0, 0),
    };
    if tensor.shape.is_empty() {
        return Extents2D::new(0, 0);
    }
    let mut cols = *tensor.shape.last().unwrap();
    let mut rows = 1usize;
    if tensor.cols_take_extra_dims {
        rows = tensor.shape[0];
        for i in 1..tensor.shape.len() - 1 {
            cols *= tensor.shape[i];
        }
    } else {
        for i in 0..tensor.shape.len() - 1 {
            rows *= tensor.shape[i];
        }
    }
    if rows == 0 || cols == 0 {
        return Extents2D::new(0, 0);
    }
    Extents2D::new(rows, cols)
}

pub fn layer_suffix(layer_idx: usize) -> String {
    format!("_{}", layer_idx)
}

pub fn strip_layer_suffix(name: &str) -> String {
    match name.rfind('_') {
        Some(pos) => name[..pos].to_string(),
        None => name.to_string(),
    }
}

pub struct TensorInfoRegistry {
    tensors: Vec<TensorInfo>,
    idx_from_name: HashMap<String, usize>,
}

impl TensorInfoRegistry {
    pub fn new(config: &ModelConfig) -> Self {
        let mut registry = Self {
            tensors: Vec::new(),
            idx_from_name: HashMap::new(),
        };
        registry.add_model_tensors(config);
        for layer_idx in 0..config.layers.len() {
            registry.add_layer_tensors(config, &config.layers[layer_idx], layer_idx);
        }
        for layer_idx in 0..config.vit_config.layer_configs.len() {
            registry.add_image_layer_tensors(
                config,
                &config.vit_config.layer_configs[layer_idx],
                layer_idx,
            );
        }
        registry
    }

    pub fn find(&self, name: &str) -> Option<&TensorInfo> {
        self.idx_from_name
            .get(name)
            .and_then(|idx| self.tensors.get(*idx))
    }

    pub fn tensor_info_from_name(&self, name: &str) -> TensorInfo {
        self.find(name).cloned().unwrap_or_default()
    }

    pub fn tensor_info_from_source_path(&self, path: &str, layer_idx: i32) -> TensorInfo {
        for info in &self.tensors {
            for source in &info.source_names {
                if path.ends_with(source) {
                    if layer_idx < 0 || info.base_name.ends_with(&layer_suffix(layer_idx as usize))
                    {
                        return info.clone();
                    }
                }
            }
        }
        TensorInfo::default()
    }

    pub fn names(&self) -> Vec<String> {
        self.idx_from_name.keys().cloned().collect()
    }

    fn add(&mut self, suffix: &str, info: &TensorInfo) {
        let mut info = info.clone();
        for name in &mut info.concat_names {
            name.push_str(suffix);
        }
        let name = format!("{}{}", info.base_name, suffix);
        let idx = self.tensors.len();
        self.tensors.push(info);
        self.idx_from_name.insert(name, idx);
    }

    fn add_model_tensors(&mut self, config: &ModelConfig) {
        let no_suffix = "";
        self.add(
            no_suffix,
            &TensorInfo {
                base_name: "c_embedding".to_string(),
                source_names: vec!["embedder/input_embedding".to_string()],
                axes: vec![0, 1],
                shape: vec![config.vocab_size as usize, config.model_dim as usize],
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
        self.add(
            no_suffix,
            &TensorInfo {
                base_name: "c_final_norm".to_string(),
                source_names: vec!["final_norm/scale".to_string()],
                axes: vec![0],
                shape: vec![config.model_dim as usize],
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
        self.add(
            no_suffix,
            &TensorInfo {
                base_name: "enc_norm_bias".to_string(),
                source_names: vec!["img/Transformer/encoder_norm/bias".to_string()],
                axes: vec![0],
                shape: vec![config.vit_config.model_dim as usize],
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
        self.add(
            no_suffix,
            &TensorInfo {
                base_name: "enc_norm_scale".to_string(),
                source_names: vec!["img/Transformer/encoder_norm/scale".to_string()],
                axes: vec![0],
                shape: vec![config.vit_config.model_dim as usize],
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
        self.add(
            no_suffix,
            &TensorInfo {
                base_name: "img_emb_bias".to_string(),
                source_names: vec!["img/embedding/bias".to_string()],
                axes: vec![0],
                shape: vec![config.vit_config.model_dim as usize],
                min_size: Type::F32,
                ..TensorInfo::default()
            },
        );
        self.add(
            no_suffix,
            &TensorInfo {
                base_name: "img_emb_kernel".to_string(),
                source_names: vec!["img/embedding/kernel".to_string()],
                axes: vec![3, 0, 1, 2],
                shape: vec![
                    config.vit_config.model_dim as usize,
                    config.vit_config.patch_width as usize,
                    config.vit_config.patch_width as usize,
                    3,
                ],
                min_size: Type::BF16,
                cols_take_extra_dims: true,
                ..TensorInfo::default()
            },
        );
        self.add(
            no_suffix,
            &TensorInfo {
                base_name: "img_head_bias".to_string(),
                source_names: vec![
                    "img/head/bias".to_string(),
                    "embedder/mm_input_projection/b".to_string(),
                ],
                axes: vec![0],
                shape: vec![config.model_dim as usize],
                min_size: Type::F32,
                ..TensorInfo::default()
            },
        );
        self.add(
            no_suffix,
            &TensorInfo {
                base_name: "img_head_kernel".to_string(),
                source_names: vec![
                    "img/head/kernel".to_string(),
                    "embedder/mm_input_projection/w".to_string(),
                ],
                axes: vec![1, 0],
                shape: vec![
                    config.model_dim as usize,
                    config.vit_config.model_dim as usize,
                ],
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
        self.add(
            no_suffix,
            &TensorInfo {
                base_name: "img_pos_emb".to_string(),
                source_names: vec!["img/pos_embedding".to_string()],
                axes: vec![0, 1],
                shape: vec![
                    config.vit_config.seq_len as usize,
                    config.vit_config.model_dim as usize,
                ],
                min_size: Type::F32,
                ..TensorInfo::default()
            },
        );
        self.add(
            no_suffix,
            &TensorInfo {
                base_name: "mm_embed_norm".to_string(),
                source_names: vec!["embedder/mm_soft_embedding_norm/scale".to_string()],
                axes: vec![0],
                shape: vec![config.vit_config.model_dim as usize],
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
    }

    fn add_image_layer_tensors(
        &mut self,
        config: &ModelConfig,
        layer: &LayerConfig,
        layer_idx: usize,
    ) {
        let suffix = layer_suffix(layer_idx);
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "attn_out_w".to_string(),
                source_names: vec!["MultiHeadDotProductAttention_0/out/kernel".to_string()],
                axes: vec![2, 0, 1],
                shape: vec![
                    config.vit_config.model_dim as usize,
                    layer.heads as usize,
                    layer.qkv_dim as usize,
                ],
                min_size: Type::BF16,
                cols_take_extra_dims: true,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "attn_out_b".to_string(),
                source_names: vec!["MultiHeadDotProductAttention_0/out/bias".to_string()],
                axes: vec![0],
                shape: vec![config.vit_config.model_dim as usize],
                min_size: Type::F32,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "q_ein_w".to_string(),
                source_names: vec!["MultiHeadDotProductAttention_0/query/kernel".to_string()],
                axes: vec![1, 2, 0],
                shape: vec![
                    layer.heads as usize,
                    layer.qkv_dim as usize,
                    config.vit_config.model_dim as usize,
                ],
                concat_names: vec![
                    "qkv_ein_w".to_string(),
                    "k_ein_w".to_string(),
                    "v_ein_w".to_string(),
                ],
                concat_axis: 1,
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "k_ein_w".to_string(),
                source_names: vec!["MultiHeadDotProductAttention_0/key/kernel".to_string()],
                axes: vec![1, 2, 0],
                shape: vec![
                    layer.heads as usize,
                    layer.qkv_dim as usize,
                    config.vit_config.model_dim as usize,
                ],
                concat_names: vec!["".to_string()],
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "v_ein_w".to_string(),
                source_names: vec!["MultiHeadDotProductAttention_0/value/kernel".to_string()],
                axes: vec![1, 2, 0],
                shape: vec![
                    layer.heads as usize,
                    layer.qkv_dim as usize,
                    config.vit_config.model_dim as usize,
                ],
                concat_names: vec!["".to_string()],
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "qkv_ein_w".to_string(),
                source_names: vec!["MultiHeadDotProductAttention_0/qkv/kernel".to_string()],
                axes: vec![1, 2, 0],
                shape: vec![
                    layer.heads as usize,
                    (3 * layer.qkv_dim) as usize,
                    config.vit_config.model_dim as usize,
                ],
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "q_ein_b".to_string(),
                source_names: vec!["MultiHeadDotProductAttention_0/query/bias".to_string()],
                axes: vec![0, 1],
                shape: vec![layer.heads as usize, layer.qkv_dim as usize],
                concat_names: vec![
                    "qkv_ein_b".to_string(),
                    "k_ein_b".to_string(),
                    "v_ein_b".to_string(),
                ],
                concat_axis: 1,
                min_size: Type::F32,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "k_ein_b".to_string(),
                source_names: vec!["MultiHeadDotProductAttention_0/key/bias".to_string()],
                axes: vec![0, 1],
                shape: vec![layer.kv_heads as usize, layer.qkv_dim as usize],
                concat_names: vec!["".to_string()],
                min_size: Type::F32,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "v_ein_b".to_string(),
                source_names: vec!["MultiHeadDotProductAttention_0/value/bias".to_string()],
                axes: vec![0, 1],
                shape: vec![layer.kv_heads as usize, layer.qkv_dim as usize],
                concat_names: vec!["".to_string()],
                min_size: Type::F32,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "qkv_ein_b".to_string(),
                source_names: vec!["MultiHeadDotProductAttention_0/qkv/bias".to_string()],
                axes: vec![0, 1],
                shape: vec![
                    (layer.heads + layer.kv_heads * 2) as usize,
                    layer.qkv_dim as usize,
                ],
                min_size: Type::F32,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "linear_0_w".to_string(),
                source_names: vec!["MlpBlock_0/Dense_0/kernel".to_string()],
                axes: vec![1, 0],
                shape: vec![
                    layer.ff_hidden_dim as usize,
                    config.vit_config.model_dim as usize,
                ],
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "linear_0_b".to_string(),
                source_names: vec!["MlpBlock_0/Dense_0/bias".to_string()],
                axes: vec![0],
                shape: vec![layer.ff_hidden_dim as usize],
                min_size: Type::F32,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "linear_1_w".to_string(),
                source_names: vec!["MlpBlock_0/Dense_1/kernel".to_string()],
                axes: vec![1, 0],
                shape: vec![
                    config.vit_config.model_dim as usize,
                    layer.ff_hidden_dim as usize,
                ],
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "linear_1_b".to_string(),
                source_names: vec!["MlpBlock_0/Dense_1/bias".to_string()],
                axes: vec![0],
                shape: vec![config.vit_config.model_dim as usize],
                min_size: Type::F32,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "ln_0_bias".to_string(),
                source_names: vec![
                    "img/Transformer/encoderblock/LayerNorm_0/bias".to_string(),
                    format!("img/Transformer/encoderblock_{layer_idx}/LayerNorm_0/bias"),
                ],
                axes: vec![0],
                shape: vec![config.vit_config.model_dim as usize],
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "ln_0_scale".to_string(),
                source_names: vec![
                    "img/Transformer/encoderblock/LayerNorm_0/scale".to_string(),
                    format!("img/Transformer/encoderblock_{layer_idx}/LayerNorm_0/scale"),
                ],
                axes: vec![0],
                shape: vec![config.vit_config.model_dim as usize],
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "ln_1_bias".to_string(),
                source_names: vec![
                    "img/Transformer/encoderblock/LayerNorm_1/bias".to_string(),
                    format!("img/Transformer/encoderblock_{layer_idx}/LayerNorm_1/bias"),
                ],
                axes: vec![0],
                shape: vec![config.vit_config.model_dim as usize],
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "ln_1_scale".to_string(),
                source_names: vec![
                    "img/Transformer/encoderblock/LayerNorm_1/scale".to_string(),
                    format!("img/Transformer/encoderblock_{layer_idx}/LayerNorm_1/scale"),
                ],
                axes: vec![0],
                shape: vec![config.vit_config.model_dim as usize],
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
    }

    fn add_layer_tensors(&mut self, config: &ModelConfig, layer: &LayerConfig, layer_idx: usize) {
        let suffix = layer_suffix(layer_idx);
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "key_norm".to_string(),
                source_names: vec!["attn/_key_norm/scale".to_string()],
                axes: vec![0],
                shape: vec![layer.qkv_dim as usize],
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "query_norm".to_string(),
                source_names: vec!["attn/_query_norm/scale".to_string()],
                axes: vec![0],
                shape: vec![layer.qkv_dim as usize],
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "qkv1_w".to_string(),
                source_names: vec!["attn/q_einsum/w".to_string()],
                axes: vec![0, 2, 1],
                shape: vec![
                    (layer.heads * layer.qkv_dim) as usize,
                    config.model_dim as usize,
                ],
                concat_names: vec!["qkv_ein".to_string(), "qkv2_w".to_string()],
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "qkv2_w".to_string(),
                source_names: vec!["attn/kv_einsum/w".to_string()],
                axes: vec![1, 0, 3, 2],
                shape: vec![
                    (2 * layer.kv_heads * layer.qkv_dim) as usize,
                    config.model_dim as usize,
                ],
                concat_names: vec!["".to_string()],
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "q_ein".to_string(),
                source_names: vec!["attention_block/proj_q/kernel".to_string()],
                axes: vec![1, 0],
                shape: vec![layer.model_dim as usize, layer.model_dim as usize],
                concat_names: vec![
                    "qkv_ein".to_string(),
                    "k_ein".to_string(),
                    "v_ein".to_string(),
                ],
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "k_ein".to_string(),
                source_names: vec!["attention_block/proj_k/kernel".to_string()],
                axes: vec![1, 0],
                shape: vec![layer.qkv_dim as usize, layer.model_dim as usize],
                concat_names: vec!["".to_string()],
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "v_ein".to_string(),
                source_names: vec!["attention_block/proj_v/kernel".to_string()],
                axes: vec![1, 0],
                shape: vec![layer.qkv_dim as usize, layer.model_dim as usize],
                concat_names: vec!["".to_string()],
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "qkv_ein".to_string(),
                source_names: vec!["attn/qkv_einsum/w".to_string()],
                axes: vec![1, 0, 3, 2],
                shape: vec![
                    ((layer.heads + 2 * layer.kv_heads) * layer.qkv_dim) as usize,
                    config.model_dim as usize,
                ],
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "attn_ob".to_string(),
                source_names: vec!["attention_block/proj_final/bias".to_string()],
                axes: vec![0],
                shape: vec![config.model_dim as usize],
                min_size: Type::F32,
                ..TensorInfo::default()
            },
        );

        self.add(
            &suffix,
            &TensorInfo {
                base_name: "gating_ein".to_string(),
                source_names: vec![
                    "mlp/gating_einsum/w".to_string(),
                    "mlp/gating_einsum".to_string(),
                    "mlp_block/ffw_up/w".to_string(),
                ],
                axes: vec![
                    0,
                    if layer.optimized_gating { 1 } else { 2 },
                    if layer.optimized_gating { 2 } else { 1 },
                ],
                shape: vec![2, layer.ff_hidden_dim as usize, config.model_dim as usize],
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "gating1_w".to_string(),
                source_names: vec!["none".to_string()],
                axes: vec![
                    0,
                    if layer.optimized_gating { 1 } else { 2 },
                    if layer.optimized_gating { 2 } else { 1 },
                ],
                shape: vec![layer.ff_hidden_dim as usize, config.model_dim as usize],
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "gating2_w".to_string(),
                source_names: vec!["none".to_string()],
                axes: vec![
                    0,
                    if layer.optimized_gating { 1 } else { 2 },
                    if layer.optimized_gating { 2 } else { 1 },
                ],
                shape: vec![layer.ff_hidden_dim as usize, config.model_dim as usize],
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "linear_w".to_string(),
                source_names: vec![
                    "mlp/linear/w".to_string(),
                    "mlp/linear".to_string(),
                    "mlp_block/ffw_down/kernel".to_string(),
                ],
                axes: vec![1, 0],
                shape: vec![config.model_dim as usize, layer.ff_hidden_dim as usize],
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "pre_att_ns".to_string(),
                source_names: vec![
                    "pre_attention_norm/scale".to_string(),
                    "temporal_pre_norm/scale".to_string(),
                ],
                axes: vec![0],
                shape: vec![config.model_dim as usize],
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "pre_ff_ns".to_string(),
                source_names: vec![
                    "pre_ffw_norm/scale".to_string(),
                    "channel_pre_norm/scale".to_string(),
                ],
                axes: vec![0],
                shape: vec![config.model_dim as usize],
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "post_att_ns".to_string(),
                source_names: vec!["post_attention_norm/scale".to_string()],
                axes: vec![0],
                shape: vec![config.model_dim as usize],
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "post_ff_ns".to_string(),
                source_names: vec!["post_ffw_norm/scale".to_string()],
                axes: vec![0],
                shape: vec![config.model_dim as usize],
                min_size: Type::BF16,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "ffw_gat_b".to_string(),
                source_names: vec!["mlp_block/ffw_up/b".to_string()],
                axes: vec![0],
                shape: vec![(2 * layer.ff_hidden_dim) as usize],
                min_size: Type::F32,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "ffw_out_b".to_string(),
                source_names: vec!["mlp_block/ffw_down/bias".to_string()],
                axes: vec![0],
                shape: vec![config.model_dim as usize],
                min_size: Type::F32,
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "att_ein".to_string(),
                source_names: vec![
                    "attn/attn_vec_einsum/w".to_string(),
                    "attention_block/proj_final/kernel".to_string(),
                ],
                preshape: vec![
                    layer.heads as usize,
                    layer.qkv_dim as usize,
                    config.model_dim as usize,
                ],
                axes: vec![0, 2, 1],
                shape: vec![
                    layer.heads as usize,
                    config.model_dim as usize,
                    layer.qkv_dim as usize,
                ],
                ..TensorInfo::default()
            },
        );
        self.add(
            &suffix,
            &TensorInfo {
                base_name: "att_w".to_string(),
                shape: vec![
                    config.model_dim as usize,
                    layer.heads as usize,
                    layer.qkv_dim as usize,
                ],
                cols_take_extra_dims: true,
                ..TensorInfo::default()
            },
        );
    }
}
