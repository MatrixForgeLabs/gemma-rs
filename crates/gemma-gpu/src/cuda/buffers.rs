//! CUDA device buffer types.

use crate::backend::{Buffer, WeightBuffer};
use cudarc::driver::safe::CudaSlice;
use gemma_compression::types::Type;

/// f32 activation buffer on CUDA device memory.
pub struct CudaBuffer {
    pub(crate) data: CudaSlice<f32>,
    pub(crate) len: usize,
}

impl Buffer for CudaBuffer {
    fn len(&self) -> usize {
        self.len
    }
}

/// Strategy used to store weights on device.
#[derive(Debug, Clone, Copy)]
pub(crate) enum WeightStorage {
    /// Weights were decompressed on host and stored as f32 on device.
    DecompressedF32,
    /// Weights are stored in their original compressed format on device.
    /// Decompression happens on-device via custom CUDA kernels.
    #[allow(dead_code)]
    Compressed,
}

/// Immutable weight matrix on CUDA device memory.
///
/// Weights may be stored as f32 (decompressed on host for Pascal) or in
/// compressed format (for Volta+ with on-device decompression kernels).
#[allow(dead_code)]
pub struct CudaWeightBuffer {
    /// The device memory holding weight data.
    /// For DecompressedF32: this is f32 elements (rows * cols).
    /// For Compressed: this is raw bytes reinterpreted as f32 for storage.
    pub(crate) data: CudaSlice<f32>,
    pub(crate) ty: Type,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) storage: WeightStorage,
    /// Number of f32 elements actually allocated.
    pub(crate) device_elements: usize,
}

impl WeightBuffer for CudaWeightBuffer {
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
        self.device_elements * std::mem::size_of::<f32>()
    }
}
