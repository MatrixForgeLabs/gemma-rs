use std::path::PathBuf;

use gemma_fetch::{huggingface, kaggle, FetchError};

fn print_usage() {
    eprintln!(
        "gemma-fetch usage:
  --kaggle <owner/model/framework/variation> [--version N] [--out path]
  --search-kaggle <query> [--limit N]
  --hf-repo <repo> --hf-file <path> [--hf-rev <rev>] [--out path]
  --search-hf <query> [--limit N]

Env:
  KAGGLE_USERNAME / KAGGLE_KEY or ~/.kaggle/kaggle.json
  HF_TOKEN or HUGGINGFACE_TOKEN (optional for private repos)"
    );
}

fn main() {
    // Load .env if present for tokens/keys.
    let _ = dotenvy::dotenv();

    let args: Vec<String> = std::env::args().collect();
    if args.len() == 1 || args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return;
    }

    let mut kaggle_handle: Option<String> = None;
    let mut kaggle_version: Option<u32> = None;
    let mut search_kaggle: Option<String> = None;
    let mut hf_repo: Option<String> = None;
    let mut hf_file: Option<String> = None;
    let mut hf_rev: Option<String> = None;
    let mut search_hf: Option<String> = None;
    let mut out: Option<PathBuf> = None;
    let mut limit: usize = 10;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--kaggle" => {
                i += 1;
                if i < args.len() {
                    kaggle_handle = Some(args[i].clone());
                }
            }
            "--version" => {
                i += 1;
                if i < args.len() {
                    kaggle_version = args[i].parse().ok();
                }
            }
            "--search-kaggle" => {
                i += 1;
                if i < args.len() {
                    search_kaggle = Some(args[i].clone());
                }
            }
            "--hf-repo" => {
                i += 1;
                if i < args.len() {
                    hf_repo = Some(args[i].clone());
                }
            }
            "--hf-file" => {
                i += 1;
                if i < args.len() {
                    hf_file = Some(args[i].clone());
                }
            }
            "--hf-rev" => {
                i += 1;
                if i < args.len() {
                    hf_rev = Some(args[i].clone());
                }
            }
            "--search-hf" => {
                i += 1;
                if i < args.len() {
                    search_hf = Some(args[i].clone());
                }
            }
            "--out" => {
                i += 1;
                if i < args.len() {
                    out = Some(PathBuf::from(&args[i]));
                }
            }
            "--limit" => {
                i += 1;
                if i < args.len() {
                    limit = args[i].parse().unwrap_or(limit);
                }
            }
            _ => {}
        }
        i += 1;
    }

    if let Some(q) = search_hf {
        match huggingface::search_models(&q, limit, std::env::var("HF_TOKEN").ok().as_deref()) {
            Ok(models) => {
                for m in models {
                    println!(
                        "{}\tdownloads={}\tlikes={}\tpipeline={}",
                        m.model_id,
                        m.downloads.unwrap_or(0),
                        m.likes.unwrap_or(0),
                        m.pipeline_tag.unwrap_or_default()
                    );
                }
            }
            Err(e) => exit_err(e),
        }
        return;
    }

    if let Some(q) = search_kaggle {
        match kaggle::search_models(&q, limit) {
            Ok(models) => {
                for m in models {
                    let fw = m.framework.clone().unwrap_or_default();
                    let subtitle = m.subtitle.clone().unwrap_or_default();
                    println!("{} / {} / {}", m.owner_slug, m.model_slug, fw);
                    if !subtitle.is_empty() {
                        println!("  {subtitle}");
                    }
                }
            }
            Err(e) => exit_err(e),
        }
        return;
    }

    if let Some(handle) = kaggle_handle {
        let handle = match kaggle::KaggleHandle::parse(&handle) {
            Ok(h) => h,
            Err(e) => return exit_err(e),
        };
        let dest = out.unwrap_or_else(|| PathBuf::from("."));
        match kaggle::download_model(&handle, kaggle_version, &dest) {
            Ok(path) => {
                println!("Downloaded to {}", path.display());
            }
            Err(e) => exit_err(e),
        }
        return;
    }

    if hf_repo.is_some() && hf_file.is_some() {
        let dest = out.unwrap_or_else(|| PathBuf::from("."));
        let token_env = std::env::var("HF_TOKEN")
            .ok()
            .or_else(|| std::env::var("HUGGINGFACE_TOKEN").ok());
        match huggingface::download_file(
            hf_repo.as_ref().unwrap(),
            hf_file.as_ref().unwrap(),
            hf_rev.as_deref(),
            token_env.as_deref(),
            &dest,
        ) {
            Ok(path) => println!("Downloaded to {}", path.display()),
            Err(e) => exit_err(e),
        }
        return;
    }

    print_usage();
}

fn exit_err(e: FetchError) {
    eprintln!("error: {e}");
    std::process::exit(1);
}
