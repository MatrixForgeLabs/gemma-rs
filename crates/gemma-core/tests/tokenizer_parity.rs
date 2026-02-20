use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use gemma_core::model_store::ModelStore;
use gemma_core::tokenizer::Tokenizer;
use gemma_io::io::Path as GemmaPath;

fn run(cmd: &mut Command) {
    let status = cmd.status().expect("failed to spawn command");
    if !status.success() {
        panic!("command failed: {cmd:?}");
    }
}

fn build_tokenizer_dump(gemma_cpp_dir: &Path) -> PathBuf {
    let build_dir = gemma_cpp_dir.join("build");
    let tool_path = build_dir.join("tokenizer_dump");
    if tool_path.exists() {
        return tool_path;
    }

    fs::create_dir_all(&build_dir).expect("failed to create build dir");
    run(Command::new("cmake").args([
        "-S",
        gemma_cpp_dir.to_str().unwrap(),
        "-B",
        build_dir.to_str().unwrap(),
        "-DSPM_ENABLE_SHARED=OFF",
        "-DSPM_ABSL_PROVIDER=module",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        "-DCMAKE_CXX_FLAGS=-DHWY_BROKEN_AVX10_2=HWY_AVX10_2",
    ]));

    let spm_header = build_dir.join("_deps/sentencepiece-src/src/sentencepiece_processor.h");
    if let Ok(contents) = fs::read_to_string(&spm_header) {
        if !contents.contains("<cstdint>") {
            let mut patched = String::new();
            for line in contents.lines() {
                patched.push_str(line);
                patched.push('\n');
                if line.trim() == "#include <vector>" {
                    patched.push_str("#include <cstdint>\n");
                }
            }
            let _ = fs::write(&spm_header, patched);
        }
    }
    run(Command::new("cmake").args([
        "--build",
        build_dir.to_str().unwrap(),
        "--target",
        "tokenizer_dump",
        "-j",
    ]));
    tool_path
}

#[test]
fn tokenizer_parity_with_gemma_cpp() {
    let gemma_cpp_dir = match std::env::var("GEMMA_CPP_DIR") {
        Ok(val) => PathBuf::from(val),
        Err(_) => {
            eprintln!("GEMMA_CPP_DIR not set; skipping tokenizer parity test");
            return;
        }
    };

    let sbs_path = if let Ok(val) = std::env::var("GEMMA_SBS_PATH") {
        PathBuf::from(val)
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../gemma-3-gemmacpp-3.0-270m-it-sfp-v1/270m-sfp-it.sbs")
    };
    if !sbs_path.exists() {
        eprintln!("SBS file not found; skipping tokenizer parity test: {sbs_path:?}");
        return;
    }

    let store = ModelStore::new(GemmaPath::new(sbs_path.to_str().unwrap()));
    let tokenizer_bytes = store
        .read_blob("tokenizer")
        .expect("tokenizer blob missing");
    let rust_tokenizer = Tokenizer::from_bytes(&tokenizer_bytes).expect("failed to load tokenizer");

    let temp_root =
        std::env::temp_dir().join(format!("gemma-tokenizer-parity-{}", std::process::id()));
    fs::create_dir_all(&temp_root).expect("failed to create temp dir");
    let tokenizer_path = temp_root.join("tokenizer.spm");
    fs::write(&tokenizer_path, &tokenizer_bytes).expect("failed to write tokenizer");

    let prompts = vec![
        "",
        "Hello",
        "Hello world",
        "Leading  space",
        "double  space",
        "Numbers 12345",
        "Punctuation:.,!?()[]{}",
        "The quick brown fox jumps over the lazy dog",
    ];
    let prompts_path = temp_root.join("prompts.txt");
    {
        let mut f = fs::File::create(&prompts_path).expect("failed to create prompts file");
        for p in &prompts {
            writeln!(f, "{p}").expect("failed to write prompt");
        }
    }

    let tool = build_tokenizer_dump(&gemma_cpp_dir);
    let output = Command::new(tool)
        .arg(&tokenizer_path)
        .arg(&prompts_path)
        .output()
        .expect("failed to run tokenizer_dump");
    if !output.status.success() {
        panic!(
            "tokenizer_dump failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(
        lines.len(),
        prompts.len(),
        "tokenizer_dump output lines mismatch"
    );

    for (idx, prompt) in prompts.iter().enumerate() {
        let rust_tokens = rust_tokenizer.encode(prompt);
        let cpp_tokens: Vec<i32> = if lines[idx].trim().is_empty() {
            Vec::new()
        } else {
            lines[idx]
                .split_whitespace()
                .map(|v| v.parse::<i32>().expect("invalid token id"))
                .collect()
        };
        assert_eq!(
            rust_tokens, cpp_tokens,
            "token mismatch at prompt index {idx}"
        );
    }
}
