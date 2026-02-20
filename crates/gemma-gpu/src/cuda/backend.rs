//! CUDA backend implementation using cudarc + optional cuBLAS.
//!
//! On GPUs where cuBLAS supports the architecture (Volta+), cuBLAS SGEMV
//! is used for matvec. On older GPUs (Pascal), a custom CUDA kernel is used.

use std::sync::{Arc, Mutex};

use cudarc::driver::result;
use cudarc::driver::safe::{CudaContext, CudaSlice, CudaStream, LaunchConfig};
use cudarc::driver::PushKernelArg;

use gemma_compression::types::Type;

use super::buffers::{CudaBuffer, CudaWeightBuffer, WeightStorage};
use super::kernels::CudaKernels;
use crate::backend::{Backend, BackendError, Buffer, DeviceCaps, OpKind, Result};

/// Optional cuBLAS handle — not available on Pascal with CUDA 13.x.
#[cfg(feature = "cuda")]
type OptionalBlas = Option<cudarc::cublas::safe::CudaBlas>;

struct ReduceScratch {
    vals: CudaSlice<f32>,
    idx: CudaSlice<i32>,
    len: usize,
}

/// CUDA backend for GPU-accelerated inference.
///
/// Uses cuBLAS for matrix-vector multiply when available (Volta+),
/// falling back to a custom CUDA kernel on older architectures.
pub struct CudaBackend {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: OptionalBlas,
    kernels: CudaKernels,
    caps: DeviceCaps,
    reduce_scratch: Mutex<Option<ReduceScratch>>,
}

// SAFETY: CudaBackend fields are all internally thread-safe.
// cudarc's CudaContext and CudaStream are Arc-wrapped and Send+Sync.
// CudaBlas handle is bound to a single stream.
// CudaKernels holds CudaFunction which is Send+Sync.
unsafe impl Send for CudaBackend {}
unsafe impl Sync for CudaBackend {}

impl CudaBackend {
    /// Create a new CUDA backend for the given device ordinal (0-indexed).
    ///
    /// cuBLAS initialization is attempted but not required — if it fails
    /// (e.g. on Pascal GPUs with CUDA 13.x), a custom matvec kernel is used.
    pub fn new(device_ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(device_ordinal)
            .map_err(|e| BackendError::DeviceError(format!("CudaContext::new: {e}")))?;

        let stream = ctx.default_stream();

        // Try to create cuBLAS; it may fail on unsupported architectures.
        let blas = match cudarc::cublas::safe::CudaBlas::new(stream.clone()) {
            Ok(b) => {
                eprintln!("GPU: cuBLAS initialized successfully");
                Some(b)
            }
            Err(e) => {
                eprintln!("GPU: cuBLAS unavailable ({e}), using custom matvec kernel");
                None
            }
        };

        let kernels = CudaKernels::load(&stream)?;

        // Query device properties.
        let name = ctx
            .name()
            .map_err(|e| BackendError::DeviceError(format!("device name: {e}")))?;

        let (compute_major, compute_minor) = ctx
            .compute_capability()
            .map_err(|e| BackendError::DeviceError(format!("compute_capability: {e}")))?;

        let (free_memory, total_memory) = result::mem_get_info()
            .map_err(|e| BackendError::DeviceError(format!("cuMemGetInfo: {e}")))?;

        let caps = DeviceCaps {
            name,
            compute_major: compute_major as u32,
            compute_minor: compute_minor as u32,
            total_memory,
            free_memory,
            has_tensor_cores: compute_major >= 7,
            has_fp16: compute_major >= 6,
            has_bf16: compute_major >= 8,
            has_fp8: compute_major >= 9,
        };

        Ok(Self {
            ctx,
            stream,
            blas,
            kernels,
            caps,
            reduce_scratch: Mutex::new(None),
        })
    }

    /// Launch configuration for a single-block kernel with shared memory.
    fn launch_single_block(&self, threads: u32, shared_mem: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: shared_mem,
        }
    }

    /// Launch configuration for a grid-stride kernel.
    fn launch_grid_stride(&self, n: usize, threads: u32) -> LaunchConfig {
        let blocks = ((n as u32) + threads - 1) / threads;
        LaunchConfig {
            grid_dim: (blocks.max(1), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Launch the custom matvec kernel: y = A * x.
    fn custom_matvec(
        &self,
        a: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let threads = 256u32;
        let shared_mem = threads * 4; // f32 per thread
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: shared_mem,
        };
        let rows_i = rows as i32;
        let cols_i = cols as i32;

        unsafe {
            self.stream
                .launch_builder(&self.kernels.matvec_f32)
                .arg(a)
                .arg(x)
                .arg(y)
                .arg(&rows_i)
                .arg(&cols_i)
                .launch(cfg)
                .map_err(|e| BackendError::KernelError(format!("custom matvec: {e}")))?;
        }
        Ok(())
    }

    /// Decompress weight data on the host to f32.
    fn decompress_to_f32(packed: &[u8], ty: Type, rows: usize, cols: usize) -> Vec<f32> {
        let n = rows * cols;
        let mut out = vec![0.0f32; n];
        match ty {
            Type::F32 => {
                let src = unsafe { std::slice::from_raw_parts(packed.as_ptr() as *const f32, n) };
                out.copy_from_slice(src);
            }
            Type::BF16 => {
                let src =
                    unsafe { std::slice::from_raw_parts(packed.as_ptr() as *const half::bf16, n) };
                for (i, &v) in src.iter().enumerate() {
                    out[i] = f32::from(v);
                }
            }
            Type::SFP => {
                for r in 0..rows {
                    let start = r * cols;
                    let end = start + cols;
                    gemma_compression::sfp::decode_f32(&packed[start..end], &mut out[start..end]);
                }
            }
            Type::NUQ => {
                let packed_nuq = unsafe {
                    std::slice::from_raw_parts(
                        packed.as_ptr() as *const gemma_compression::types::NuqStream,
                        packed.len(),
                    )
                };
                for r in 0..rows {
                    gemma_compression::nuq::decompress_and_zero_pad_f32(
                        packed_nuq,
                        r * cols,
                        &mut out[r * cols..(r + 1) * cols],
                        cols,
                    );
                }
            }
            Type::I8 => {
                let packed_i8 = unsafe {
                    std::slice::from_raw_parts(
                        packed.as_ptr() as *const gemma_compression::types::I8Stream,
                        packed.len(),
                    )
                };
                for r in 0..rows {
                    gemma_compression::int_format::decompress_and_zero_pad_f32(
                        packed_i8,
                        r * cols,
                        &mut out[r * cols..(r + 1) * cols],
                        cols,
                    );
                }
            }
            _ => panic!("upload_weight: unsupported type {:?}", ty),
        }
        out
    }
}

impl Backend for CudaBackend {
    type Buf = CudaBuffer;
    type Wgt = CudaWeightBuffer;

    fn caps(&self) -> &DeviceCaps {
        &self.caps
    }

    fn alloc(&self, len: usize) -> Result<CudaBuffer> {
        let data: CudaSlice<f32> =
            self.stream
                .alloc_zeros(len)
                .map_err(|_| BackendError::OutOfMemory {
                    requested: len * 4,
                    available: self.caps.free_memory,
                })?;
        Ok(CudaBuffer { data, len })
    }

    fn upload_weight(
        &self,
        packed: &[u8],
        ty: Type,
        rows: usize,
        cols: usize,
    ) -> Result<CudaWeightBuffer> {
        // Phase 2: always decompress on host and upload f32.
        let f32_data = Self::decompress_to_f32(packed, ty, rows, cols);
        let device_elements = f32_data.len();

        let mut data = self
            .stream
            .alloc_zeros::<f32>(device_elements)
            .map_err(|e| BackendError::TransferError(format!("alloc for upload_weight: {e}")))?;

        self.stream
            .memcpy_htod(&f32_data, &mut data)
            .map_err(|e| BackendError::TransferError(format!("upload_weight: {e}")))?;

        Ok(CudaWeightBuffer {
            data,
            ty,
            rows,
            cols,
            storage: WeightStorage::DecompressedF32,
            device_elements,
        })
    }

    fn upload_f32(&self, src: &[f32], dst: &mut CudaBuffer) -> Result<()> {
        if src.len() != dst.len {
            return Err(BackendError::TransferError(format!(
                "upload_f32 size mismatch: src={}, dst={}",
                src.len(),
                dst.len
            )));
        }
        self.stream
            .memcpy_htod(src, &mut dst.data)
            .map_err(|e| BackendError::TransferError(format!("upload_f32: {e}")))?;
        Ok(())
    }

    fn download_f32(&self, src: &CudaBuffer, dst: &mut [f32]) -> Result<()> {
        if src.len != dst.len() {
            return Err(BackendError::TransferError(format!(
                "download_f32 size mismatch: src={}, dst={}",
                src.len,
                dst.len()
            )));
        }
        self.stream
            .memcpy_dtoh(&src.data, dst)
            .map_err(|e| BackendError::TransferError(format!("download_f32: {e}")))?;
        Ok(())
    }

    fn matvec(&self, weight: &CudaWeightBuffer, x: &CudaBuffer, y: &mut CudaBuffer) -> Result<()> {
        let rows = weight.rows;
        let cols = weight.cols;

        if let Some(ref blas) = self.blas {
            // cuBLAS SGEMV: y = alpha * op(A) * x + beta * y
            // Our weights are row-major: W[i][j] = data[i * cols + j].
            // cuBLAS sees this as a column-major matrix of shape (cols × rows).
            // To compute y = W * x (row-major), we compute y = A^T * x (column-major).
            use cudarc::cublas::safe::GemvConfig;
            use cudarc::cublas::sys::cublasOperation_t;
            use cudarc::cublas::Gemv;

            let cfg = GemvConfig {
                trans: cublasOperation_t::CUBLAS_OP_T,
                m: cols as i32,
                n: rows as i32,
                alpha: 1.0f32,
                lda: cols as i32,
                incx: 1,
                beta: 0.0f32,
                incy: 1,
            };

            unsafe {
                blas.gemv(cfg, &weight.data, &x.data, &mut y.data)
                    .map_err(|e| BackendError::KernelError(format!("cublas sgemv: {e}")))?;
            }
        } else {
            // Custom kernel fallback for Pascal and other unsupported archs.
            self.custom_matvec(&weight.data, &x.data, &mut y.data, rows, cols)?;
        }

        Ok(())
    }

    fn matvec_head(
        &self,
        weight: &CudaWeightBuffer,
        head: usize,
        _model_dim: usize,
        qkv_dim: usize,
        x: &CudaBuffer,
        y: &mut CudaBuffer,
    ) -> Result<()> {
        let cols = weight.cols;
        let row_start = head * qkv_dim;

        let weight_slice = weight
            .data
            .try_slice(row_start * cols..(row_start + qkv_dim) * cols)
            .ok_or_else(|| BackendError::KernelError("weight slice out of bounds".into()))?;

        let mut y_slice = y
            .data
            .try_slice_mut(0..qkv_dim)
            .ok_or_else(|| BackendError::KernelError("y slice out of bounds".into()))?;

        if let Some(ref blas) = self.blas {
            use cudarc::cublas::safe::GemvConfig;
            use cudarc::cublas::sys::cublasOperation_t;
            use cudarc::cublas::Gemv;

            let cfg = GemvConfig {
                trans: cublasOperation_t::CUBLAS_OP_T,
                m: cols as i32,
                n: qkv_dim as i32,
                alpha: 1.0f32,
                lda: cols as i32,
                incx: 1,
                beta: 0.0f32,
                incy: 1,
            };

            unsafe {
                blas.gemv(cfg, &weight_slice, &x.data, &mut y_slice)
                    .map_err(|e| BackendError::KernelError(format!("cublas sgemv head: {e}")))?;
            }
        } else {
            // Inline kernel launch for view types (CudaView/CudaViewMut).
            let threads = 256u32;
            let shared_mem = threads * 4;
            let cfg = LaunchConfig {
                grid_dim: (qkv_dim as u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: shared_mem,
            };
            let rows_i = qkv_dim as i32;
            let cols_i = cols as i32;
            unsafe {
                self.stream
                    .launch_builder(&self.kernels.matvec_f32)
                    .arg(&weight_slice)
                    .arg(&x.data)
                    .arg(&mut y_slice)
                    .arg(&rows_i)
                    .arg(&cols_i)
                    .launch(cfg)
                    .map_err(|e| BackendError::KernelError(format!("custom matvec head: {e}")))?;
            }
        }

        Ok(())
    }

    fn rms_norm(&self, values: &mut CudaBuffer, scale: &CudaBuffer, eps: f32) -> Result<()> {
        let n = values.len as i32;
        let threads = 256u32;
        let shared_mem = threads * 4;
        let cfg = self.launch_single_block(threads, shared_mem);

        unsafe {
            self.stream
                .launch_builder(&self.kernels.rms_norm)
                .arg(&mut values.data)
                .arg(&scale.data)
                .arg(&n)
                .arg(&eps)
                .launch(cfg)
                .map_err(|e| BackendError::KernelError(format!("rms_norm: {e}")))?;
        }
        Ok(())
    }

    fn softmax(&self, values: &mut CudaBuffer) -> Result<()> {
        let n = values.len as i32;
        let threads = 256u32;
        let shared_mem = threads * 4;
        let cfg = self.launch_single_block(threads, shared_mem);

        unsafe {
            self.stream
                .launch_builder(&self.kernels.softmax)
                .arg(&mut values.data)
                .arg(&n)
                .launch(cfg)
                .map_err(|e| BackendError::KernelError(format!("softmax: {e}")))?;
        }
        Ok(())
    }

    fn rope_inplace(&self, values: &mut CudaBuffer, pos: f32) -> Result<()> {
        let rope_dim = values.len as i32;
        let half = values.len / 2;
        let cfg = self.launch_grid_stride(half, 256);

        unsafe {
            self.stream
                .launch_builder(&self.kernels.rope_inplace)
                .arg(&mut values.data)
                .arg(&rope_dim)
                .arg(&pos)
                .launch(cfg)
                .map_err(|e| BackendError::KernelError(format!("rope: {e}")))?;
        }
        Ok(())
    }

    fn dot(&self, a: &CudaBuffer, b: &CudaBuffer) -> Result<f32> {
        // For small vectors, download and compute on CPU.
        let mut a_host = vec![0.0f32; a.len];
        let mut b_host = vec![0.0f32; b.len];
        self.download_f32(a, &mut a_host)?;
        self.download_f32(b, &mut b_host)?;
        Ok(gemma_ops::dot::dot_f32(&a_host, &b_host))
    }

    fn gelu_gate(&self, src: &CudaBuffer, hidden_dim: usize, dst: &mut CudaBuffer) -> Result<()> {
        let hd = hidden_dim as i32;
        let cfg = self.launch_grid_stride(hidden_dim, 256);

        unsafe {
            self.stream
                .launch_builder(&self.kernels.gelu_gate)
                .arg(&src.data)
                .arg(&mut dst.data)
                .arg(&hd)
                .launch(cfg)
                .map_err(|e| BackendError::KernelError(format!("gelu_gate: {e}")))?;
        }
        Ok(())
    }

    fn add_inplace(&self, dst: &mut CudaBuffer, src: &CudaBuffer) -> Result<()> {
        let n = dst.len as i32;
        let cfg = self.launch_grid_stride(dst.len, 256);

        unsafe {
            self.stream
                .launch_builder(&self.kernels.add_inplace)
                .arg(&mut dst.data)
                .arg(&src.data)
                .arg(&n)
                .launch(cfg)
                .map_err(|e| BackendError::KernelError(format!("add_inplace: {e}")))?;
        }
        Ok(())
    }

    fn scale_inplace(&self, dst: &mut CudaBuffer, factor: f32) -> Result<()> {
        let n = dst.len as i32;
        let cfg = self.launch_grid_stride(dst.len, 256);

        unsafe {
            self.stream
                .launch_builder(&self.kernels.scale_inplace)
                .arg(&mut dst.data)
                .arg(&factor)
                .arg(&n)
                .launch(cfg)
                .map_err(|e| BackendError::KernelError(format!("scale_inplace: {e}")))?;
        }
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| BackendError::DeviceError(format!("synchronize: {e}")))?;
        Ok(())
    }

    fn argmax(&self, buf: &Self::Buf) -> Result<usize> {
        self.argmax_with_scratch(buf, None)
    }

    fn argmax_with_scratch(
        &self,
        buf: &Self::Buf,
        _scratch: Option<&mut Self::Buf>,
    ) -> Result<usize> {
        let n = buf.len();
        if n == 0 {
            return Err(BackendError::Unsupported(
                "argmax on empty buffer".to_string(),
            ));
        }

        // Allocate (or reuse) a single scratch pair sized to the block count.
        let threads = 256u32;
        let blocks = ((n as u32 + threads - 1) / threads) as usize;
        let mut scratch_guard = self.reduce_scratch.lock().unwrap();
        if scratch_guard
            .as_ref()
            .map_or(true, |s| s.len < blocks)
        {
            let vals = self
                .stream
                .alloc_zeros::<f32>(blocks)
                .map_err(|_| BackendError::OutOfMemory {
                    requested: blocks * std::mem::size_of::<f32>(),
                    available: self.caps.free_memory,
                })?;
            let idx = self
                .stream
                .alloc_zeros::<i32>(blocks)
                .map_err(|_| BackendError::OutOfMemory {
                    requested: blocks * std::mem::size_of::<i32>(),
                    available: self.caps.free_memory,
                })?;
            *scratch_guard = Some(ReduceScratch { vals, idx, len: blocks });
        }

        let scratch = scratch_guard.as_mut().unwrap();
        let n_i32 = n as i32;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.kernels.reduce_block_max)
                .arg(&buf.data)
                .arg(&n_i32)
                .arg(&mut scratch.vals)
                .arg(&mut scratch.idx)
                .launch(cfg)
                .map_err(|e| BackendError::KernelError(format!("reduce_block_max: {e}")))?;
        }

        // Ensure kernel completion before copying block maxima back.
        self.stream
            .synchronize()
            .map_err(|e| BackendError::DeviceError(format!("argmax sync: {e}")))?;

        let mut block_vals = vec![0.0f32; blocks];
        let mut block_idx = vec![0i32; blocks];
        self.stream
            .memcpy_dtoh(&scratch.vals, &mut block_vals)
            .map_err(|e| BackendError::TransferError(format!("argmax dtoh vals: {e}")))?;
        self.stream
            .memcpy_dtoh(&scratch.idx, &mut block_idx)
            .map_err(|e| BackendError::TransferError(format!("argmax dtoh idx: {e}")))?;

        let mut best = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for b in 0..blocks {
            let v = block_vals[b];
            let i = block_idx[b];
            if i >= 0 && v > best_val {
                best_val = v;
                best = i as usize;
            }
        }
        Ok(best)
    }

    fn supports_op(&self, op: OpKind) -> bool {
        match op {
            OpKind::FlashAttention => false,
            _ => true,
        }
    }
}
