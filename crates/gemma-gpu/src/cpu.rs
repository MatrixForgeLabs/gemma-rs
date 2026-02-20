//! CPU backend implementation wrapping existing gemma-ops functions.

use crate::backend::{Backend, BackendError, Buffer, DeviceCaps, OpKind, Result, WeightBuffer};
use gemma_compression::types::Type;

// ── Buffer types ────────────────────────────────────────────────────

/// CPU activation buffer backed by a `Vec<f32>`.
pub struct CpuBuffer {
    data: Vec<f32>,
}

impl CpuBuffer {
    /// View the buffer contents as a slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// View the buffer contents as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }
}

impl Buffer for CpuBuffer {
    fn len(&self) -> usize {
        self.data.len()
    }
}

/// CPU weight buffer storing raw packed bytes with matrix metadata.
pub struct CpuWeightBuffer {
    data: Vec<u8>,
    ty: Type,
    rows: usize,
    cols: usize,
}

impl CpuWeightBuffer {
    /// Access the packed weight bytes.
    pub fn packed(&self) -> &[u8] {
        &self.data
    }
}

impl WeightBuffer for CpuWeightBuffer {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn ty(&self) -> Type {
        self.ty
    }

    fn size_bytes(&self) -> usize {
        self.data.len()
    }
}

// ── Backend ─────────────────────────────────────────────────────────

/// CPU backend that delegates to gemma-ops scalar/AVX2 implementations.
pub struct CpuBackend {
    caps: DeviceCaps,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            caps: DeviceCaps {
                name: "CPU".to_string(),
                compute_major: 0,
                compute_minor: 0,
                total_memory: usize::MAX,
                free_memory: usize::MAX,
                has_tensor_cores: false,
                has_fp16: false,
                has_bf16: false,
                has_fp8: false,
            },
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    type Buf = CpuBuffer;
    type Wgt = CpuWeightBuffer;

    fn caps(&self) -> &DeviceCaps {
        &self.caps
    }

    fn alloc(&self, len: usize) -> Result<CpuBuffer> {
        Ok(CpuBuffer {
            data: vec![0.0f32; len],
        })
    }

    fn upload_weight(
        &self,
        packed: &[u8],
        ty: Type,
        rows: usize,
        cols: usize,
    ) -> Result<CpuWeightBuffer> {
        Ok(CpuWeightBuffer {
            data: packed.to_vec(),
            ty,
            rows,
            cols,
        })
    }

    fn upload_f32(&self, src: &[f32], dst: &mut CpuBuffer) -> Result<()> {
        if src.len() != dst.data.len() {
            return Err(BackendError::TransferError(format!(
                "upload_f32 size mismatch: src={}, dst={}",
                src.len(),
                dst.data.len()
            )));
        }
        dst.data.copy_from_slice(src);
        Ok(())
    }

    fn download_f32(&self, src: &CpuBuffer, dst: &mut [f32]) -> Result<()> {
        if src.data.len() != dst.len() {
            return Err(BackendError::TransferError(format!(
                "download_f32 size mismatch: src={}, dst={}",
                src.data.len(),
                dst.len()
            )));
        }
        dst.copy_from_slice(&src.data);
        Ok(())
    }

    fn matvec(&self, weight: &CpuWeightBuffer, x: &CpuBuffer, y: &mut CpuBuffer) -> Result<()> {
        gemma_ops::matmul::matvec_dispatch(
            weight.ty,
            &weight.data,
            weight.rows,
            weight.cols,
            &x.data,
            &mut y.data,
        );
        Ok(())
    }

    fn matvec_head(
        &self,
        weight: &CpuWeightBuffer,
        head: usize,
        _model_dim: usize,
        qkv_dim: usize,
        x: &CpuBuffer,
        y: &mut CpuBuffer,
    ) -> Result<()> {
        // Per-head projection: extract the head's rows from the weight matrix
        // and multiply against the input vector.
        //
        // The weight matrix has shape (model_dim × cols), where each head owns
        // `qkv_dim` contiguous rows starting at `head * qkv_dim`.
        let row_start = head * qkv_dim;
        let row_end = row_start + qkv_dim;
        if row_end > weight.rows {
            return Err(BackendError::KernelError(format!(
                "matvec_head: head {} exceeds weight rows ({}×{})",
                head, weight.rows, weight.cols
            )));
        }

        // For F32 weights, we can slice directly. For compressed types,
        // we decompress the full matrix and slice -- this is the CPU fallback
        // so simplicity over performance is fine here.
        let ty = weight.ty;
        let cols = weight.cols;

        match ty {
            Type::F32 => {
                let elem_size = std::mem::size_of::<f32>();
                let row_bytes = cols * elem_size;
                let start = row_start * row_bytes;
                let end = row_end * row_bytes;
                let sub_packed = &weight.data[start..end];
                gemma_ops::matmul::matvec_dispatch(
                    ty,
                    sub_packed,
                    qkv_dim,
                    cols,
                    &x.data,
                    &mut y.data[..qkv_dim],
                );
            }
            Type::BF16 => {
                let elem_size = std::mem::size_of::<half::bf16>();
                let row_bytes = cols * elem_size;
                let start = row_start * row_bytes;
                let end = row_end * row_bytes;
                let sub_packed = &weight.data[start..end];
                gemma_ops::matmul::matvec_dispatch(
                    ty,
                    sub_packed,
                    qkv_dim,
                    cols,
                    &x.data,
                    &mut y.data[..qkv_dim],
                );
            }
            _ => {
                // For compressed types, fall back to full matvec and extract the head's rows.
                // This is wasteful but correct -- GPU backend will do better.
                let mut full_y = vec![0.0f32; weight.rows];
                gemma_ops::matmul::matvec_dispatch(
                    ty,
                    &weight.data,
                    weight.rows,
                    cols,
                    &x.data,
                    &mut full_y,
                );
                y.data[..qkv_dim].copy_from_slice(&full_y[row_start..row_end]);
            }
        }

        Ok(())
    }

    fn rms_norm(&self, values: &mut CpuBuffer, scale: &CpuBuffer, eps: f32) -> Result<()> {
        gemma_ops::nn::rms_norm(&mut values.data, &scale.data, eps);
        Ok(())
    }

    fn softmax(&self, values: &mut CpuBuffer) -> Result<()> {
        gemma_ops::nn::softmax(&mut values.data);
        Ok(())
    }

    fn rope_inplace(&self, values: &mut CpuBuffer, pos: f32) -> Result<()> {
        gemma_ops::nn::rope_inplace(&mut values.data, pos);
        Ok(())
    }

    fn dot(&self, a: &CpuBuffer, b: &CpuBuffer) -> Result<f32> {
        Ok(gemma_ops::dot::dot_f32(&a.data, &b.data))
    }

    fn gelu_gate(&self, src: &CpuBuffer, hidden_dim: usize, dst: &mut CpuBuffer) -> Result<()> {
        if src.data.len() != 2 * hidden_dim {
            return Err(BackendError::KernelError(format!(
                "gelu_gate: src len {} != 2 * hidden_dim {}",
                src.data.len(),
                hidden_dim
            )));
        }
        let (gate, up) = src.data.split_at(hidden_dim);
        for i in 0..hidden_dim {
            dst.data[i] = gelu(gate[i]) * up[i];
        }
        Ok(())
    }

    fn add_inplace(&self, dst: &mut CpuBuffer, src: &CpuBuffer) -> Result<()> {
        for i in 0..dst.data.len() {
            dst.data[i] += src.data[i];
        }
        Ok(())
    }

    fn scale_inplace(&self, dst: &mut CpuBuffer, factor: f32) -> Result<()> {
        for v in dst.data.iter_mut() {
            *v *= factor;
        }
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        // CPU is synchronous -- no-op.
        Ok(())
    }

    fn supports_op(&self, _op: OpKind) -> bool {
        true
    }
}

/// GELU activation function (same as gemma-core's implementation).
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
}
