//! Model configuration structures (partial port).

use gemma_compression::types::Type;
use gemma_io::fields::{Fields, FieldsVisitor};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PromptWrapping {
    GemmaIt,
    GemmaPt,
    GemmaVlm,
    PaliGemma,
}

impl Default for PromptWrapping {
    fn default() -> Self {
        PromptWrapping::GemmaPt
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LayerAttentionType {
    Gemma,
    Vit,
}

impl Default for LayerAttentionType {
    fn default() -> Self {
        LayerAttentionType::Gemma
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PostNormType {
    None,
    Scale,
}

impl Default for PostNormType {
    fn default() -> Self {
        PostNormType::None
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PostQKType {
    Rope,
    HalfRope,
}

impl Default for PostQKType {
    fn default() -> Self {
        PostQKType::Rope
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ActivationType {
    Gelu,
}

impl Default for ActivationType {
    fn default() -> Self {
        ActivationType::Gelu
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum QueryScaleType {
    SqrtKeySize,
    SqrtModelDimDivNumHeads,
}

impl Default for QueryScaleType {
    fn default() -> Self {
        QueryScaleType::SqrtKeySize
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ResidualType {
    Add,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Model {
    Unknown,
    Gemma2_9B,
    Gemma2_27B,
    Gemma2_2B,
    PaliGemma2_3b224,
    PaliGemma2_3b448,
    PaliGemma2_10b224,
    PaliGemma2_10b448,
    Gemma3_4B,
    Gemma3_1B,
    Gemma3_12B,
    Gemma3_27B,
    Gemma3_270M,
}

impl Default for Model {
    fn default() -> Self {
        Model::Unknown
    }
}

#[derive(Clone, Debug, Default)]
pub struct LayerConfig {
    pub layer_idx: usize,
    pub attention_type: LayerAttentionType,
    pub post_norm: PostNormType,
    pub use_qk_norm: bool,
    pub ff_biases: bool,
    pub model_dim: u32,
    pub ff_hidden_dim: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub qkv_dim: u32,
    pub optimized_gating: bool,
    pub activation: ActivationType,
    pub post_qk: PostQKType,
}

#[derive(Clone, Debug, Default)]
pub struct ModelConfig {
    pub model: Model,
    pub wrapping: PromptWrapping,
    pub num_layers: usize,
    pub has_vision: bool,
    pub layers: Vec<LayerConfig>,
    pub weight: Type,
    pub model_dim: u32,
    pub vocab_size: u32,
    pub max_seq_len: u32,
    pub att_cap: f32,
    pub final_cap: f32,
    pub absolute_pe: bool,
    pub query_scale: QueryScaleType,
    pub attention_window_sizes: Vec<u32>,
    pub norm_num_groups: u32,
    pub vit_config: VitConfig,
    pub pool_dim: u32,
    pub eos_id: i32,
    pub secondary_eos_id: i32,
    pub scale_base_names: Vec<String>,
}

impl ModelConfig {
    pub fn new(num_layers: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for idx in 0..num_layers {
            layers.push(LayerConfig {
                layer_idx: idx,
                attention_type: LayerAttentionType::Gemma,
                post_norm: PostNormType::None,
                use_qk_norm: false,
                ff_biases: false,
                model_dim: 0,
                ff_hidden_dim: 0,
                heads: 0,
                kv_heads: 0,
                qkv_dim: 0,
                optimized_gating: true,
                activation: ActivationType::Gelu,
                post_qk: PostQKType::Rope,
            });
        }
        Self {
            model: Model::Unknown,
            wrapping: PromptWrapping::GemmaPt,
            num_layers,
            has_vision: false,
            layers,
            weight: Type::Unknown,
            model_dim: 0,
            vocab_size: 0,
            max_seq_len: 0,
            att_cap: 0.0,
            final_cap: 0.0,
            absolute_pe: false,
            query_scale: QueryScaleType::SqrtKeySize,
            attention_window_sizes: Vec::new(),
            norm_num_groups: 1,
            vit_config: VitConfig::default(),
            pool_dim: 1,
            eos_id: 1,
            secondary_eos_id: 1,
            scale_base_names: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VitConfig {
    pub model_dim: u32,
    pub seq_len: u32,
    pub num_scales: u32,
    pub patch_width: u32,
    pub image_size: u32,
    pub pool_dim: u32,
    pub layer_configs: Vec<LayerConfig>,
}

#[derive(Clone, Debug, Default)]
pub struct InternalLayerConfig;

#[derive(Clone, Debug, Default)]
pub struct InternalModelConfig;

impl Fields for LayerConfig {
    fn name(&self) -> &str {
        "LayerConfig"
    }

    fn visit_fields(&mut self, visitor: &mut dyn FieldsVisitor) {
        let mut unused_griffin_dim: u32 = 0;
        let mut unused_conv1d_width: u32 = 0;
        let mut unused_softmax_attn_output_biases: bool = false;

        visitor.visit_u32(&mut self.model_dim);
        visitor.visit_u32(&mut unused_griffin_dim);
        visitor.visit_u32(&mut self.ff_hidden_dim);
        visitor.visit_u32(&mut self.heads);
        visitor.visit_u32(&mut self.kv_heads);
        visitor.visit_u32(&mut self.qkv_dim);
        visitor.visit_u32(&mut unused_conv1d_width);
        visitor.visit_bool(&mut self.ff_biases);
        visitor.visit_bool(&mut unused_softmax_attn_output_biases);
        visitor.visit_bool(&mut self.optimized_gating);

        let mut post_norm = self.post_norm as u32;
        visitor.visit_u32(&mut post_norm);
        self.post_norm = match post_norm {
            1 => PostNormType::Scale,
            _ => PostNormType::None,
        };

        let mut att_type = self.attention_type as u32;
        visitor.visit_u32(&mut att_type);
        self.attention_type = match att_type {
            1 => LayerAttentionType::Vit,
            _ => LayerAttentionType::Gemma,
        };

        let mut activation = self.activation as u32;
        visitor.visit_u32(&mut activation);
        self.activation = ActivationType::Gelu;

        let mut post_qk = self.post_qk as u32;
        visitor.visit_u32(&mut post_qk);
        self.post_qk = match post_qk {
            1 => PostQKType::HalfRope,
            _ => PostQKType::Rope,
        };

        visitor.visit_bool(&mut self.use_qk_norm);
    }
}

impl Fields for VitConfig {
    fn name(&self) -> &str {
        "VitConfig"
    }

    fn visit_fields(&mut self, visitor: &mut dyn FieldsVisitor) {
        visitor.visit_u32(&mut self.model_dim);
        visitor.visit_u32(&mut self.seq_len);
        visitor.visit_u32(&mut self.num_scales);
        visitor.visit_u32(&mut self.patch_width);
        visitor.visit_u32(&mut self.image_size);

        let mut num_layers = self.layer_configs.len() as u32;
        visitor.visit_u32(&mut num_layers);
        if visitor.is_reading() {
            self.layer_configs
                .resize_with(num_layers as usize, LayerConfig::default);
        }
        for layer in &mut self.layer_configs {
            visitor.visit_fields(layer);
        }

        visitor.visit_u32(&mut self.pool_dim);
    }
}

impl Fields for ModelConfig {
    fn name(&self) -> &str {
        "ModelConfig"
    }

    fn visit_fields(&mut self, visitor: &mut dyn FieldsVisitor) {
        let mut model_family_version: u32 = 1;
        let mut display_name = String::new();

        visitor.visit_u32(&mut model_family_version);
        visitor.visit_string(&mut display_name);

        let mut model = self.model as u32;
        visitor.visit_u32(&mut model);
        self.model = match model {
            3 => Model::Gemma2_9B,
            4 => Model::Gemma2_27B,
            7 => Model::Gemma2_2B,
            10 => Model::PaliGemma2_3b224,
            11 => Model::PaliGemma2_3b448,
            12 => Model::PaliGemma2_10b224,
            13 => Model::PaliGemma2_10b448,
            14 => Model::Gemma3_4B,
            15 => Model::Gemma3_1B,
            16 => Model::Gemma3_12B,
            17 => Model::Gemma3_27B,
            18 => Model::Gemma3_270M,
            _ => Model::Unknown,
        };

        let mut wrapping = self.wrapping as u32;
        visitor.visit_u32(&mut wrapping);
        self.wrapping = match wrapping {
            0 => PromptWrapping::GemmaIt,
            1 => PromptWrapping::GemmaPt,
            2 => PromptWrapping::GemmaVlm,
            3 => PromptWrapping::PaliGemma,
            _ => PromptWrapping::GemmaPt,
        };

        let mut weight = self.weight as u32;
        visitor.visit_u32(&mut weight);
        self.weight = match weight {
            1 => Type::F32,
            2 => Type::BF16,
            3 => Type::SFP,
            4 => Type::NUQ,
            5 => Type::F64,
            6 => Type::U32,
            7 => Type::U64,
            8 => Type::I8,
            _ => Type::Unknown,
        };

        let mut num_layers = self.num_layers as u32;
        visitor.visit_u32(&mut num_layers);
        self.num_layers = num_layers as usize;

        visitor.visit_u32(&mut self.model_dim);
        visitor.visit_u32(&mut self.vocab_size);
        visitor.visit_u32(&mut self.max_seq_len);

        let mut unused_num_tensor_scales: u32 = 0;
        visitor.visit_u32(&mut unused_num_tensor_scales);

        visitor.visit_f32(&mut self.att_cap);
        visitor.visit_f32(&mut self.final_cap);

        visitor.visit_bool(&mut self.absolute_pe);
        let mut unused_use_local_attention: bool = false;
        visitor.visit_bool(&mut unused_use_local_attention);

        let mut query_scale = self.query_scale as u32;
        visitor.visit_u32(&mut query_scale);
        self.query_scale = match query_scale {
            1 => QueryScaleType::SqrtModelDimDivNumHeads,
            _ => QueryScaleType::SqrtKeySize,
        };

        let mut layer_len = self.layers.len() as u32;
        visitor.visit_u32(&mut layer_len);
        if visitor.is_reading() {
            self.layers
                .resize_with(layer_len as usize, LayerConfig::default);
        }
        for layer in &mut self.layers {
            visitor.visit_fields(layer);
        }

        visitor.visit_vec_u32(&mut self.attention_window_sizes);
        visitor.visit_u32(&mut self.norm_num_groups);
        visitor.visit_fields(&mut self.vit_config);
        visitor.visit_u32(&mut self.pool_dim);

        visitor.visit_i32(&mut self.eos_id);
        visitor.visit_i32(&mut self.secondary_eos_id);

        visitor.visit_vec_string(&mut self.scale_base_names);
    }
}
