use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use gemma_core::configs::ModelConfig;
use gemma_core::gemma::{Gemma, SamplingOptions};

const EX_USAGE: i32 = 64;
const EX_IOERR: i32 = 74;
const EX_UNAVAILABLE: i32 = 69;
const EXIT_MISMATCH: i32 = 2;

#[derive(Debug)]
struct Args {
    weights: PathBuf,
    prompts: PathBuf,
    cpp_bin: PathBuf,
    report: PathBuf,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: Option<u64>,
}

#[derive(Debug)]
struct PromptResult {
    prompt: String,
    rust_out: String,
    cpp_out: String,
    matched: bool,
}

fn main() {
    match run() {
        Ok(()) => {}
        Err((code, msg)) => {
            eprintln!("{msg}");
            std::process::exit(code);
        }
    }
}

fn run() -> Result<(), (i32, String)> {
    let args = parse_args()?;
    validate_inputs(&args)?;

    let prompts = load_prompts(&args.prompts)?;
    let config = ModelConfig::new(0);
    let model = Gemma::from_sbs(config, args.weights.to_string_lossy().as_ref());
    let sampling = SamplingOptions {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        seed: args.seed,
    };

    let mut results = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let rust_out = model.generate_with_sampling(&prompt, args.max_tokens, sampling);
        let cpp_out = run_cpp_generation(&args, &prompt)?;
        let matched = rust_out == cpp_out;
        results.push(PromptResult {
            prompt,
            rust_out,
            cpp_out,
            matched,
        });
    }

    write_report(&args, &results)?;

    let mismatches = results.iter().filter(|r| !r.matched).count();
    println!(
        "Parity run complete: total={}, matches={}, mismatches={}, report={}",
        results.len(),
        results.len() - mismatches,
        mismatches,
        args.report.display()
    );

    if mismatches > 0 {
        return Err((EXIT_MISMATCH, format!("Found {mismatches} mismatches")));
    }

    Ok(())
}

fn parse_args() -> Result<Args, (i32, String)> {
    let mut weights = env::var("GEMMA_SBS_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../gemma-3-gemmacpp-3.0-270m-it-sfp-v1/270m-sfp-it.sbs")
        });
    let mut prompts =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/parity/prompts.txt");
    let mut cpp_bin = env::var("GEMMA_CPP_BIN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp/gemma-cpp-build-rust-parity/gemma"));
    let mut report = default_report_path();
    let mut max_tokens = 32usize;
    let mut temperature = 1.0f32;
    let mut top_k = 1usize;
    let mut top_p = 1.0f32;
    let mut seed = None;

    let mut iter = env::args().skip(1);
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--weights" => weights = parse_path_flag("--weights", iter.next())?,
            "--prompts" => prompts = parse_path_flag("--prompts", iter.next())?,
            "--cpp-bin" => cpp_bin = parse_path_flag("--cpp-bin", iter.next())?,
            "--report" => report = parse_path_flag("--report", iter.next())?,
            "--max-tokens" => max_tokens = parse_num_flag("--max-tokens", iter.next())?,
            "--temperature" => temperature = parse_num_flag("--temperature", iter.next())?,
            "--top-k" => top_k = parse_num_flag("--top-k", iter.next())?,
            "--top-p" => top_p = parse_num_flag("--top-p", iter.next())?,
            "--seed" => seed = Some(parse_num_flag::<u64>("--seed", iter.next())?),
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {
                return Err((
                    EX_USAGE,
                    format!("Unknown flag '{flag}'. Use --help for usage."),
                ))
            }
        }
    }

    Ok(Args {
        weights,
        prompts,
        cpp_bin,
        report,
        max_tokens,
        temperature,
        top_k,
        top_p,
        seed,
    })
}

fn parse_path_flag(flag: &str, value: Option<String>) -> Result<PathBuf, (i32, String)> {
    value
        .map(PathBuf::from)
        .ok_or_else(|| (EX_USAGE, format!("Missing value for {flag}")))
}

fn parse_num_flag<T: std::str::FromStr>(
    flag: &str,
    value: Option<String>,
) -> Result<T, (i32, String)> {
    let raw = value.ok_or_else(|| (EX_USAGE, format!("Missing value for {flag}")))?;
    raw.parse::<T>()
        .map_err(|_| (EX_USAGE, format!("Invalid value for {flag}: {raw}")))
}

fn print_help() {
    println!(
        "gemma-evals text-parity\n\n\
Usage:\n\
  cargo run -p gemma-evals -- [options]\n\n\
Options:\n\
  --weights <path>      Path to .sbs model (default: GEMMA_SBS_PATH or local 270M)\n\
  --prompts <path>      Prompt file, one prompt per line (default: scripts/parity/prompts.txt)\n\
  --cpp-bin <path>      gemma.cpp binary path (default: /tmp/gemma-cpp-build-rust-parity/gemma)\n\
  --report <path>       Markdown output path (default: reports/parity/text-parity-<ts>.md)\n\
  --max-tokens <n>      Generated tokens per prompt (default: 32)\n\
  --temperature <f32>   Sampling temperature (default: 1.0)\n\
  --top-k <n>           Top-k (default: 1)\n\
  --top-p <f32>         Top-p (default: 1.0)\n\
  --seed <u64>          RNG seed (optional)\n"
    );
}

fn default_report_path() -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../../reports/parity/text-parity-{ts}.md"))
}

fn validate_inputs(args: &Args) -> Result<(), (i32, String)> {
    if !args.weights.is_file() {
        return Err((
            EX_IOERR,
            format!("Missing weights file: {}", args.weights.display()),
        ));
    }
    if !args.prompts.is_file() {
        return Err((
            EX_IOERR,
            format!("Missing prompts file: {}", args.prompts.display()),
        ));
    }
    if !args.cpp_bin.is_file() {
        return Err((
            EX_UNAVAILABLE,
            format!(
                "Missing gemma.cpp binary: {} (set --cpp-bin or GEMMA_CPP_BIN)",
                args.cpp_bin.display()
            ),
        ));
    }
    Ok(())
}

fn load_prompts(path: &Path) -> Result<Vec<String>, (i32, String)> {
    let content = fs::read_to_string(path).map_err(|e| {
        (
            EX_IOERR,
            format!("Failed to read prompts '{}': {e}", path.display()),
        )
    })?;
    let prompts: Vec<String> = content
        .lines()
        .map(str::trim_end)
        .filter(|l| !l.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    if prompts.is_empty() {
        return Err((
            EX_USAGE,
            format!("Prompt file '{}' has no non-empty lines", path.display()),
        ));
    }
    Ok(prompts)
}

fn run_cpp_generation(args: &Args, prompt: &str) -> Result<String, (i32, String)> {
    let output = Command::new(&args.cpp_bin)
        .arg("--weights")
        .arg(&args.weights)
        .arg("--prompt")
        .arg(prompt)
        .arg("--max_generated_tokens")
        .arg(args.max_tokens.to_string())
        .arg("--temperature")
        .arg(args.temperature.to_string())
        .arg("--top_k")
        .arg(args.top_k.to_string())
        .arg("--deterministic")
        .arg("1")
        .arg("--verbosity")
        .arg("0")
        .output()
        .map_err(|e| {
            (
                EX_UNAVAILABLE,
                format!(
                    "Failed to spawn gemma.cpp binary '{}': {e}",
                    args.cpp_bin.display()
                ),
            )
        })?;

    if !output.status.success() {
        return Err((
            EX_UNAVAILABLE,
            format!(
                "gemma.cpp generation failed with status {}: {}",
                output
                    .status
                    .code()
                    .map_or_else(|| "signal".to_string(), |c| c.to_string()),
                String::from_utf8_lossy(&output.stderr)
            ),
        ));
    }

    Ok(String::from_utf8_lossy(&output.stdout)
        .trim_end()
        .to_string())
}

fn write_report(args: &Args, results: &[PromptResult]) -> Result<(), (i32, String)> {
    if let Some(parent) = args.report.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            (
                EX_IOERR,
                format!(
                    "Failed to create report directory '{}': {e}",
                    parent.display()
                ),
            )
        })?;
    }

    let matches = results.iter().filter(|r| r.matched).count();
    let mismatches = results.len().saturating_sub(matches);

    let mut out = String::new();
    out.push_str("# Text Parity Report\n\n");
    out.push_str("Settings\n");
    out.push_str(&format!("- weights: {}\n", args.weights.display()));
    out.push_str(&format!("- prompts: {}\n", args.prompts.display()));
    out.push_str(&format!("- cpp_bin: {}\n", args.cpp_bin.display()));
    out.push_str(&format!("- max tokens: {}\n", args.max_tokens));
    out.push_str(&format!(
        "- sampling: temperature={}, top_k={}, top_p={}, seed={:?}\n\n",
        args.temperature, args.top_k, args.top_p, args.seed
    ));

    for (idx, res) in results.iter().enumerate() {
        let status = if res.matched { "MATCH" } else { "MISMATCH" };
        out.push_str(&format!("## Prompt {} [{}]\n\n", idx + 1, status));
        out.push_str("Prompt:\n```text\n");
        out.push_str(&res.prompt);
        out.push_str("\n```\n\n");
        out.push_str("Rust:\n```text\n");
        out.push_str(&res.rust_out);
        out.push_str("\n```\n\n");
        out.push_str("C++:\n```text\n");
        out.push_str(&res.cpp_out);
        out.push_str("\n```\n\n");
    }

    out.push_str("## Summary\n\n");
    out.push_str(&format!("- total: {}\n", results.len()));
    out.push_str(&format!("- matches: {}\n", matches));
    out.push_str(&format!("- mismatches: {}\n", mismatches));

    fs::write(&args.report, out).map_err(|e| {
        (
            EX_IOERR,
            format!("Failed to write report '{}': {e}", args.report.display()),
        )
    })
}
