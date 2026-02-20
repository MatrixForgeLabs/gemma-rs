//! CUDA kernel loading via runtime compilation (NVRTC).
//!
//! Kernel sources are embedded at build time via `include_str!` and compiled
//! to PTX at runtime using `cudarc::nvrtc::compile_ptx`.  This ensures the
//! generated PTX matches the driver version on the host machine.

use crate::backend::{BackendError, Result};
use cudarc::driver::safe::{CudaFunction, CudaStream};
use cudarc::nvrtc::safe::compile_ptx;
use std::sync::Arc;

/// Holds references to all loaded CUDA kernel functions.
pub(crate) struct CudaKernels {
    pub rms_norm: CudaFunction,
    pub softmax: CudaFunction,
    pub rope_inplace: CudaFunction,
    pub gelu_gate: CudaFunction,
    pub add_inplace: CudaFunction,
    pub scale_inplace: CudaFunction,
    #[allow(dead_code)]
    pub decompress_sfp: CudaFunction,
    pub matvec_f32: CudaFunction,
    pub reduce_block_max: CudaFunction,
}

impl CudaKernels {
    /// Compile all kernel sources via NVRTC and extract kernel functions.
    pub fn load(stream: &Arc<CudaStream>) -> Result<Self> {
        let ctx = stream.context();

        // Kernel sources embedded from .cu files at compile time.
        let rms_norm_src = include_str!("rms_norm.cu");
        let softmax_src = include_str!("softmax.cu");
        let rope_src = include_str!("rope.cu");
        let gelu_gate_src = include_str!("gelu_gate.cu");
        let elementwise_src = include_str!("elementwise.cu");
        let decompress_src = include_str!("decompress.cu");
        let matvec_src = include_str!("matvec.cu");
        let reduce_src = include_str!("reduce_max.cu");

        let compile_and_load = |cu_src: &str, name: &str| -> Result<CudaFunction> {
            let ptx = compile_ptx(cu_src)
                .map_err(|e| BackendError::DeviceError(format!("nvrtc compile {name}: {e}")))?;
            let module = ctx
                .load_module(ptx)
                .map_err(|e| BackendError::DeviceError(format!("load_module({name}): {e}")))?;
            module
                .load_function(name)
                .map_err(|e| BackendError::DeviceError(format!("load_function({name}): {e}")))
        };

        Ok(Self {
            rms_norm: compile_and_load(rms_norm_src, "rms_norm")?,
            softmax: compile_and_load(softmax_src, "softmax")?,
            rope_inplace: compile_and_load(rope_src, "rope_inplace")?,
            gelu_gate: compile_and_load(gelu_gate_src, "gelu_gate")?,
            add_inplace: compile_and_load(elementwise_src, "add_inplace")?,
            scale_inplace: compile_and_load(elementwise_src, "scale_inplace")?,
            decompress_sfp: compile_and_load(decompress_src, "decompress_sfp")?,
            matvec_f32: compile_and_load(matvec_src, "matvec_f32")?,
            reduce_block_max: compile_and_load(reduce_src, "reduce_block_max")?,
        })
    }
}
