use std::path::PathBuf;

use gemma_core::model_store::ModelStore;
use gemma_core::tensor_info::{extents_from_info, TensorInfoRegistry};
use gemma_io::io::Path as GemmaPath;

#[test]
fn tensor_registry_matches_toc_shapes() {
    let sbs_path = if let Ok(val) = std::env::var("GEMMA_SBS_PATH") {
        PathBuf::from(val)
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../gemma-3-gemmacpp-3.0-270m-it-sfp-v1/270m-sfp-it.sbs")
    };
    if !sbs_path.exists() {
        eprintln!("SBS file not found; skipping tensor registry test: {sbs_path:?}");
        return;
    }

    let store = ModelStore::new(GemmaPath::new(sbs_path.to_str().unwrap()));
    let Some(config) = store.config() else {
        eprintln!("Model config not found; skipping tensor registry test");
        return;
    };
    let registry = TensorInfoRegistry::new(config);

    for mat in store.mat_ptrs() {
        let info = registry.find(mat.name());
        assert!(info.is_some(), "missing tensor info for {}", mat.name());
        let extents = extents_from_info(info);
        if extents.rows == 0 && extents.cols == 0 {
            continue;
        }
        assert_eq!(
            (extents.rows, extents.cols),
            (mat.rows(), mat.cols()),
            "shape mismatch for {}",
            mat.name()
        );
    }
}
