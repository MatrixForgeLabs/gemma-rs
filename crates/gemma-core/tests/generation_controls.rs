use std::path::PathBuf;

use gemma_core::configs::ModelConfig;
use gemma_core::gemma::{Gemma, SamplingOptions};

fn default_sbs_path() -> PathBuf {
    if let Ok(val) = std::env::var("GEMMA_SBS_PATH") {
        PathBuf::from(val)
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../gemma-3-gemmacpp-3.0-270m-it-sfp-v1/270m-sfp-it.sbs")
    }
}

#[test]
fn max_tokens_caps_output_growth() {
    let sbs_path = default_sbs_path();
    if !sbs_path.exists() {
        eprintln!("SBS file not found; skipping generation control test: {sbs_path:?}");
        return;
    }

    let config = ModelConfig::new(0);
    let model = Gemma::from_sbs(config, sbs_path.to_str().unwrap());
    let prompt = "Hello";

    let prompt_tokens = model.tokenizer.encode(prompt);
    let out_1 = model.generate(prompt, 1);
    let out_3 = model.generate(prompt, 3);
    let out_1_tokens = model.tokenizer.encode(&out_1);
    let out_3_tokens = model.tokenizer.encode(&out_3);

    assert!(out_1_tokens.len() <= prompt_tokens.len() + 1);
    assert!(out_3_tokens.len() <= prompt_tokens.len() + 3);
}

#[test]
fn eos_id_can_stop_generation_early() {
    let sbs_path = default_sbs_path();
    if !sbs_path.exists() {
        eprintln!("SBS file not found; skipping generation control test: {sbs_path:?}");
        return;
    }

    let config = ModelConfig::new(0);
    let mut model = Gemma::from_sbs(config, sbs_path.to_str().unwrap());
    let prompt = "Hello";
    let prompt_tokens = model.tokenizer.encode(prompt);

    let baseline = model.generate(prompt, 1);
    let baseline_tokens = model.tokenizer.encode(&baseline);
    assert!(baseline_tokens.len() > prompt_tokens.len());

    let generated = baseline_tokens[prompt_tokens.len()];
    model.config.eos_id = generated;
    model.config.secondary_eos_id = i32::MIN;

    let stopped = model.generate(prompt, 5);
    let stopped_tokens = model.tokenizer.encode(&stopped);
    assert!(stopped_tokens.len() <= prompt_tokens.len() + 1);
}

#[test]
fn prefix_end_none_matches_default_generate() {
    let sbs_path = default_sbs_path();
    if !sbs_path.exists() {
        eprintln!("SBS file not found; skipping generation control test: {sbs_path:?}");
        return;
    }

    let config = ModelConfig::new(0);
    let model = Gemma::from_sbs(config, sbs_path.to_str().unwrap());
    let prompt = "Hello";

    let baseline = model.generate(prompt, 2);
    let via_opts = model.generate_with_prefix_end(prompt, 2, None);
    assert_eq!(baseline, via_opts);
}

#[test]
fn prefix_end_option_generates_without_overgrowth() {
    let sbs_path = default_sbs_path();
    if !sbs_path.exists() {
        eprintln!("SBS file not found; skipping generation control test: {sbs_path:?}");
        return;
    }

    let config = ModelConfig::new(0);
    let model = Gemma::from_sbs(config, sbs_path.to_str().unwrap());
    let prompt = "Hello";
    let prompt_tokens = model.tokenizer.encode(prompt);
    let prefix_end = Some(prompt_tokens.len());

    let out = model.generate_with_prefix_end(prompt, 2, prefix_end);
    let out_tokens = model.tokenizer.encode(&out);
    assert!(out_tokens.len() <= prompt_tokens.len() + 2);
}

#[test]
fn sampling_seed_is_repeatable() {
    let sbs_path = default_sbs_path();
    if !sbs_path.exists() {
        eprintln!("SBS file not found; skipping generation control test: {sbs_path:?}");
        return;
    }

    let config = ModelConfig::new(0);
    let model = Gemma::from_sbs(config, sbs_path.to_str().unwrap());
    let prompt = "Hello";
    let sampling = SamplingOptions {
        temperature: 0.8,
        top_k: 8,
        top_p: 0.95,
        seed: Some(42),
    };

    let out_a = model.generate_with_sampling(prompt, 4, sampling);
    let out_b = model.generate_with_sampling(prompt, 4, sampling);
    assert_eq!(out_a, out_b);
}

#[test]
fn prefix_end_zero_matches_default_generate() {
    let sbs_path = default_sbs_path();
    if !sbs_path.exists() {
        eprintln!("SBS file not found; skipping generation control test: {sbs_path:?}");
        return;
    }

    let config = ModelConfig::new(0);
    let model = Gemma::from_sbs(config, sbs_path.to_str().unwrap());
    let prompt = "Hello";

    let baseline = model.generate(prompt, 3);
    let with_zero = model.generate_with_prefix_end(prompt, 3, Some(0));
    assert_eq!(baseline, with_zero);
}

#[test]
fn very_large_prefix_end_is_clamped_and_matches_default() {
    let sbs_path = default_sbs_path();
    if !sbs_path.exists() {
        eprintln!("SBS file not found; skipping generation control test: {sbs_path:?}");
        return;
    }

    let config = ModelConfig::new(0);
    let model = Gemma::from_sbs(config, sbs_path.to_str().unwrap());
    let prompt = "Hello";

    let baseline = model.generate(prompt, 3);
    let with_large = model.generate_with_prefix_end(prompt, 3, Some(usize::MAX));
    assert_eq!(baseline, with_large);
}
