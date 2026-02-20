//! Build script for gemma-gpu.
//!
//! CUDA kernels are now compiled at runtime via NVRTC (see kernels/mod.rs),
//! so this build script no longer invokes nvcc.

fn main() {
    // Rerun if any kernel source changes (triggers Rust recompile so
    // include_str! picks up updated .cu files).
    #[cfg(feature = "cuda")]
    {
        let kernel_dir = std::path::Path::new("src/cuda/kernels");
        for entry in std::fs::read_dir(kernel_dir).expect("cannot read kernel dir") {
            let entry = entry.expect("cannot read dir entry");
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "cu") {
                println!("cargo:rerun-if-changed={}", path.display());
            }
        }
    }
}
