//! GPU backend abstraction for gemma-rs inference.
//!
//! Provides a `Backend` trait that abstracts over compute devices (CPU, CUDA, etc.)
//! with associated buffer types for type-safe device memory management.

pub mod backend;
pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;
