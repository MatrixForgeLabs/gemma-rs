//! GPU backend abstraction for gemma-rs inference.
//!
//! This crate is a thin wrapper around `inference-gpu`, adding Gemma-specific
//! helpers for weight decompression (SFP, NUQ, I8) and type conversion.
//!
//! All core backend types, traits, and implementations are re-exported from
//! `inference-gpu` so that consumers can continue using `gemma_gpu::backend::*`
//! and `gemma_gpu::cuda::*` paths.

/// Re-exported backend trait, types, and error definitions.
pub use inference_gpu::backend;

/// Re-exported CPU backend implementation.
pub use inference_gpu::cpu;

/// Re-exported CUDA backend implementation.
#[cfg(feature = "cuda")]
pub use inference_gpu::cuda;

/// Gemma-specific helpers for weight format conversion and upload.
pub mod gemma;
