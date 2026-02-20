use std::path::PathBuf;

use gemma_core::configs::ModelConfig;
use gemma_core::gemma::Gemma;

#[test]
fn minimal_inference_runs() {
    let sbs_path = if let Ok(val) = std::env::var("GEMMA_SBS_PATH") {
        PathBuf::from(val)
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../gemma-3-gemmacpp-3.0-270m-it-sfp-v1/270m-sfp-it.sbs")
    };
    if !sbs_path.exists() {
        eprintln!("SBS file not found; skipping minimal inference test: {sbs_path:?}");
        return;
    }

    let config = ModelConfig::new(0);
    let model = Gemma::from_sbs(config, sbs_path.to_str().unwrap());
    let out = model.generate("Hello", 1);
    assert!(!out.is_empty());
}

#[test]
fn minimal_inference_perf_sanity() {
    if std::env::var("GEMMA_PERF_CHECK").ok().as_deref() != Some("1") {
        eprintln!("GEMMA_PERF_CHECK not set; skipping perf sanity test");
        return;
    }

    let sbs_path = if let Ok(val) = std::env::var("GEMMA_SBS_PATH") {
        PathBuf::from(val)
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../gemma-3-gemmacpp-3.0-270m-it-sfp-v1/270m-sfp-it.sbs")
    };
    if !sbs_path.exists() {
        eprintln!("SBS file not found; skipping perf sanity test: {sbs_path:?}");
        return;
    }

    let config = ModelConfig::new(0);
    let model = Gemma::from_sbs(config, sbs_path.to_str().unwrap());
    let max_ms: u128 = std::env::var("GEMMA_PERF_MAX_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(60_000);

    let start = std::time::Instant::now();
    let _ = model.generate("Hello", 1);
    let elapsed = start.elapsed().as_millis();
    assert!(
        elapsed <= max_ms,
        "minimal inference took {elapsed}ms, max {max_ms}ms"
    );
}

#[test]
fn full_forward_perf_sanity() {
    if std::env::var("GEMMA_PERF_CHECK").ok().as_deref() != Some("1") {
        eprintln!("GEMMA_PERF_CHECK not set; skipping full forward perf test");
        return;
    }
    if std::env::var("GEMMA_PERF_FULL").ok().as_deref() != Some("1") {
        eprintln!("GEMMA_PERF_FULL not set; skipping full forward perf test");
        return;
    }

    let sbs_path = if let Ok(val) = std::env::var("GEMMA_SBS_PATH") {
        PathBuf::from(val)
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../gemma-3-gemmacpp-3.0-270m-it-sfp-v1/270m-sfp-it.sbs")
    };
    if !sbs_path.exists() {
        eprintln!("SBS file not found; skipping full forward perf test: {sbs_path:?}");
        return;
    }

    let config = ModelConfig::new(0);
    let model = Gemma::from_sbs(config, sbs_path.to_str().unwrap());
    let max_ms: u128 = std::env::var("GEMMA_PERF_FULL_MAX_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(180_000);

    let start = std::time::Instant::now();
    let _ = model.generate("Hello world", 1);
    let elapsed = start.elapsed().as_millis();
    assert!(
        elapsed <= max_ms,
        "full forward took {elapsed}ms, max {max_ms}ms"
    );
}
