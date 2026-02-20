//! Backend trait and core types for device-agnostic inference.

use gemma_compression::types::Type;

/// Errors that can occur during backend operations.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    #[error("out of memory: requested {requested} bytes, {available} available")]
    OutOfMemory { requested: usize, available: usize },

    #[error("kernel error: {0}")]
    KernelError(String),

    #[error("unsupported operation: {0}")]
    Unsupported(String),

    #[error("transfer error: {0}")]
    TransferError(String),

    #[error("device error: {0}")]
    DeviceError(String),
}

pub type Result<T> = std::result::Result<T, BackendError>;

/// Device capability information, queried at backend construction time.
#[derive(Debug, Clone)]
pub struct DeviceCaps {
    pub name: String,
    /// CUDA compute capability major version (0 for CPU).
    pub compute_major: u32,
    /// CUDA compute capability minor version.
    pub compute_minor: u32,
    /// Total device memory in bytes.
    pub total_memory: usize,
    /// Free device memory in bytes (snapshot at query time).
    pub free_memory: usize,
    pub has_tensor_cores: bool,
    pub has_fp16: bool,
    pub has_bf16: bool,
    pub has_fp8: bool,
}

/// Operation kinds for capability queries via `Backend::supports_op`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpKind {
    Matvec,
    MatvecHead,
    RmsNorm,
    Softmax,
    Rope,
    Dot,
    GeluGate,
    AddInplace,
    ScaleInplace,
    FlashAttention,
}

/// An f32 activation buffer on a compute device.
///
/// Buffers are mutable working memory for intermediate activations.
/// The length is in f32 elements, not bytes.
pub trait Buffer: Send + Sized {
    /// Number of f32 elements in this buffer.
    fn len(&self) -> usize;

    /// Size in bytes.
    fn size_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<f32>()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// An immutable weight matrix on a compute device.
///
/// Weight buffers hold model parameters that are uploaded once and read many times.
/// They may be stored in compressed format (SFP, NUQ, I8) on devices that support
/// on-device decompression, or pre-decompressed to f32 on older hardware.
pub trait WeightBuffer: Send + Sized {
    /// Number of rows in the weight matrix.
    fn rows(&self) -> usize;

    /// Number of columns in the weight matrix.
    fn cols(&self) -> usize;

    /// The compression type of the stored data.
    fn ty(&self) -> Type;

    /// Size in bytes on device.
    fn size_bytes(&self) -> usize;
}

/// A compute backend for inference operations.
///
/// Implementations provide device-specific buffer allocation, data transfer,
/// and kernel execution. The trait uses associated types rather than trait objects
/// so that the hot path (matvec called thousands of times per generation) is
/// monomorphized at compile time.
///
/// # Execution model
///
/// Operations execute eagerly. On GPU backends, operations are enqueued on a
/// CUDA stream and execute asynchronously relative to the host. Call
/// `synchronize()` to wait for all enqueued operations to complete.
pub trait Backend: Send + Sync + Sized {
    /// Activation buffer type for this backend.
    type Buf: Buffer;

    /// Weight matrix type for this backend.
    type Wgt: WeightBuffer;

    /// Query device capabilities.
    fn caps(&self) -> &DeviceCaps;

    /// Allocate a zero-initialized f32 buffer with `len` elements.
    fn alloc(&self, len: usize) -> Result<Self::Buf>;

    /// Upload a weight matrix to the device.
    ///
    /// The backend chooses the optimal storage strategy based on device capabilities:
    /// - Pascal (compute 6.x): decompresses on host, uploads f32
    /// - Volta+ (compute >= 7.0): uploads compressed, decompresses on-device
    fn upload_weight(&self, packed: &[u8], ty: Type, rows: usize, cols: usize)
        -> Result<Self::Wgt>;

    /// Copy f32 data from host to a device buffer.
    ///
    /// `src` and `dst` must have the same length.
    fn upload_f32(&self, src: &[f32], dst: &mut Self::Buf) -> Result<()>;

    /// Copy f32 data from a device buffer to host.
    ///
    /// `src` and `dst` must have the same length.
    fn download_f32(&self, src: &Self::Buf, dst: &mut [f32]) -> Result<()>;

    // ── Compute operations ──────────────────────────────────────────

    /// Matrix-vector multiply: `y = weight * x`.
    ///
    /// `weight` is (rows × cols), `x` has `cols` elements, `y` has `rows` elements.
    fn matvec(&self, weight: &Self::Wgt, x: &Self::Buf, y: &mut Self::Buf) -> Result<()>;

    /// Per-head matrix-vector multiply for attention output projection.
    ///
    /// Multiplies a single head's slice of the weight matrix by the input.
    fn matvec_head(
        &self,
        weight: &Self::Wgt,
        head: usize,
        model_dim: usize,
        qkv_dim: usize,
        x: &Self::Buf,
        y: &mut Self::Buf,
    ) -> Result<()>;

    /// In-place RMS normalization: `values = rms_norm(values, scale, eps)`.
    fn rms_norm(&self, values: &mut Self::Buf, scale: &Self::Buf, eps: f32) -> Result<()>;

    /// In-place softmax over the buffer.
    fn softmax(&self, values: &mut Self::Buf) -> Result<()>;

    /// In-place rotary position embedding.
    fn rope_inplace(&self, values: &mut Self::Buf, pos: f32) -> Result<()>;

    /// Dot product of two equal-length buffers.
    fn dot(&self, a: &Self::Buf, b: &Self::Buf) -> Result<f32>;

    /// Fused GELU-gated linear unit.
    ///
    /// Input `src` has `2 * hidden_dim` elements: first half is gate, second half is up.
    /// Output `dst` has `hidden_dim` elements: `dst[i] = gelu(gate[i]) * up[i]`.
    fn gelu_gate(&self, src: &Self::Buf, hidden_dim: usize, dst: &mut Self::Buf) -> Result<()>;

    /// In-place elementwise addition: `dst += src`.
    fn add_inplace(&self, dst: &mut Self::Buf, src: &Self::Buf) -> Result<()>;

    /// In-place elementwise scaling: `dst *= factor`.
    fn scale_inplace(&self, dst: &mut Self::Buf, factor: f32) -> Result<()>;

    /// Block until all enqueued operations on this backend have completed.
    ///
    /// No-op for synchronous backends (CPU).
    fn synchronize(&self) -> Result<()>;

    /// Argmax over a buffer, returning the index of the maximum element.
    ///
    /// Used by GPU-side sampling to avoid downloading full logits.
    fn argmax(&self, buf: &Self::Buf) -> Result<usize>;

    /// Optional argmax that writes intermediate maxima to a provided scratch buffer.
    /// Implementations may ignore the scratch buffer and fall back to `argmax`.
    fn argmax_with_scratch(&self, buf: &Self::Buf, _scratch: Option<&mut Self::Buf>) -> Result<usize> {
        self.argmax(buf)
    }

    /// Query whether this backend supports a given operation kind.
    ///
    /// Used by the execution planner to decide whether to fall back to CPU
    /// for unsupported operations.
    fn supports_op(&self, op: OpKind) -> bool;
}
