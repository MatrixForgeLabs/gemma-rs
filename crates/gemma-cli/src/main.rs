use gemma_core::configs::ModelConfig;
use gemma_core::gemma::{Gemma, SamplingOptions};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print_usage();
        return;
    }
    let mut prompt = "hello".to_string();
    let mut weights = "weights.sbs".to_string();
    let mut max_tokens = 16usize;
    let mut temperature = 1.0f32;
    let mut top_k = 1usize;
    let mut top_p = 1.0f32;
    let mut seed = None::<u64>;
    let mut encode = None::<String>;
    let mut decode = None::<String>;
    let mut print_backend = false;
    let mut print_config = false;
    let mut device = "cpu".to_string();
    let mut chat = false;
    let mut stream = false;
    let mut bench = false;
    let mut bench_iters: usize = 3;
    let mut bench_warmup: usize = 1;
    let mut bench_json: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => {
                i += 1;
                if i < args.len() {
                    weights = args[i].clone();
                }
            }
            "--prompt" => {
                i += 1;
                if i < args.len() {
                    prompt = args[i].clone();
                }
            }
            "--max-tokens" => {
                i += 1;
                if i < args.len() {
                    max_tokens = args[i].parse().unwrap_or(max_tokens);
                }
            }
            "--temperature" => {
                i += 1;
                if i < args.len() {
                    temperature = args[i].parse().unwrap_or(temperature);
                }
            }
            "--top-k" => {
                i += 1;
                if i < args.len() {
                    top_k = args[i].parse().unwrap_or(top_k);
                }
            }
            "--top-p" => {
                i += 1;
                if i < args.len() {
                    top_p = args[i].parse().unwrap_or(top_p);
                }
            }
            "--seed" => {
                i += 1;
                if i < args.len() {
                    seed = args[i].parse::<u64>().ok();
                }
            }
            "--encode" => {
                i += 1;
                if i < args.len() {
                    encode = Some(args[i].clone());
                }
            }
            "--decode" => {
                i += 1;
                if i < args.len() {
                    decode = Some(args[i].clone());
                }
            }
            "--print-backend" => {
                print_backend = true;
            }
            "--print-config" => {
                print_config = true;
            }
            "--device" => {
                i += 1;
                if i < args.len() {
                    device = args[i].clone();
                }
            }
            "--chat" => {
                chat = true;
            }
            "--stream" => {
                stream = true;
            }
            "--bench" => {
                bench = true;
            }
            "--bench-iters" => {
                i += 1;
                if i < args.len() {
                    bench_iters = args[i].parse().unwrap_or(bench_iters);
                }
            }
            "--bench-warmup" => {
                i += 1;
                if i < args.len() {
                    bench_warmup = args[i].parse().unwrap_or(bench_warmup);
                }
            }
            "--bench-json" => {
                i += 1;
                if i < args.len() {
                    bench_json = Some(args[i].clone());
                }
            }
            _ => {}
        }
        i += 1;
    }

    let config = ModelConfig::new(0);
    let model = Gemma::from_sbs(config, &weights);
    if print_backend {
        println!(
            "backend: full={}, minimal={}",
            model.has_full_weights(),
            model.has_minimal_weights()
        );
    }
    if print_config {
        println!(
            "config: model={:?} layers={} model_dim={} vocab={} max_seq_len={} att_cap={} final_cap={} query_scale={:?}",
            model.config.model,
            model.config.layers.len(),
            model.config.model_dim,
            model.config.vocab_size,
            model.config.max_seq_len,
            model.config.att_cap,
            model.config.final_cap,
            model.config.query_scale
        );
        if let Some(layer0) = model.config.layers.first() {
            println!(
                "layer0: type={:?} post_norm={:?} use_qk_norm={} ff_biases={} heads={} kv_heads={} qkv_dim={} ff_hidden_dim={} optimized_gating={} activation={:?} post_qk={:?}",
                layer0.attention_type,
                layer0.post_norm,
                layer0.use_qk_norm,
                layer0.ff_biases,
                layer0.heads,
                layer0.kv_heads,
                layer0.qkv_dim,
                layer0.ff_hidden_dim,
                layer0.optimized_gating,
                layer0.activation,
                layer0.post_qk
            );
        }
    }

    if let Some(text) = encode {
        let tokens = model.tokenizer.encode(&text);
        let line = tokens
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        println!("{}", line);
        return;
    }

    if let Some(ids) = decode {
        let tokens: Vec<i32> = ids
            .split(|c| c == ',' || c == ' ' || c == '\t')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<i32>().unwrap_or(0))
            .collect();
        let text = model.tokenizer.decode(&tokens);
        println!("{}", text);
        return;
    }

    let sampling = SamplingOptions {
        temperature,
        top_k,
        top_p,
        seed,
    };

    if bench {
        match device.as_str() {
            "cpu" => run_bench_cpu(
                &model,
                &prompt,
                max_tokens,
                sampling,
                bench_warmup,
                bench_iters,
                bench_json.as_deref(),
            ),
            #[cfg(feature = "cuda")]
            d if d == "cuda" || d.starts_with("cuda:") => {
                run_bench_cpu(
                    &model,
                    &prompt,
                    max_tokens,
                    sampling,
                    bench_warmup,
                    bench_iters,
                    bench_json.as_deref(),
                );
                run_bench_cuda(
                    &model,
                    &prompt,
                    max_tokens,
                    sampling,
                    d,
                    bench_warmup,
                    bench_iters,
                    bench_json.as_deref(),
                );
            }
            #[cfg(not(feature = "cuda"))]
            d if d == "cuda" || d.starts_with("cuda:") => {
                eprintln!("CUDA support not compiled. Rebuild with: cargo build --features cuda");
                std::process::exit(1);
            }
            other => {
                eprintln!("Unknown device for bench: {other}");
                std::process::exit(1);
            }
        }
        return;
    }

    if chat {
        match device.as_str() {
            "cpu" => run_chat_cpu(&model, max_tokens, sampling, stream),
            #[cfg(feature = "cuda")]
            d if d == "cuda" || d.starts_with("cuda:") => {
                run_chat_cuda(&model, max_tokens, sampling, d, stream);
            }
            #[cfg(not(feature = "cuda"))]
            d if d == "cuda" || d.starts_with("cuda:") => {
                eprintln!("CUDA support not compiled. Rebuild with: cargo build --features cuda");
                std::process::exit(1);
            }
            other => {
                eprintln!("Unknown device: {other}. Use 'cpu' or 'cuda'");
                std::process::exit(1);
            }
        }
        return;
    }

    match device.as_str() {
        "cpu" => {
            let out = model.generate_with_sampling(&prompt, max_tokens, sampling);
            println!("{}", out);
        }
        #[cfg(feature = "cuda")]
        d if d == "cuda" || d.starts_with("cuda:") => {
            run_cuda(&model, &prompt, max_tokens, sampling, d);
        }
        #[cfg(not(feature = "cuda"))]
        d if d == "cuda" || d.starts_with("cuda:") => {
            eprintln!("CUDA support not compiled. Rebuild with: cargo build --features cuda");
            std::process::exit(1);
        }
        other => {
            eprintln!("Unknown device: {other}. Use 'cpu' or 'cuda'");
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    println!(
        "\
gemma-cli usage:
  --weights <path>      Path to weights.sbs
  --prompt <text>       Single-shot prompt (default: \"hello\")
  --max-tokens <n>      Max tokens to generate (default 16)
  --temperature <f>     Sampling temperature (default 1.0)
  --top-k <n>           Top-K sampling (default 1)
  --top-p <f>           Top-P sampling (default 1.0)
  --seed <u64>          RNG seed
  --encode <text>       Encode text to token IDs
  --decode \"1 2 3\"     Decode token IDs to text
  --print-backend       Print weights availability (full/minimal)
  --print-config        Print model config summary
  --device <cpu|cuda[:i]>  Select backend (default cpu)
  --chat                Interactive chat loop
  --stream              Stream tokens during chat
  --bench               Run quick tok/s benchmark (CPU and CUDA if selected)
  --bench-iters <n>     Measured iterations for bench (default 3)
  --bench-warmup <n>    Warmup iterations (default 1)
  --bench-json <path>   Write bench results as JSON
"
    );
}

// ── Chat helpers ────────────────────────────────────────────────────────

/// A single turn in the conversation.
struct Turn {
    user: String,
    assistant: String,
}

/// Build the full conversation string with Gemma turn markers.
///
/// Format:
///   <start_of_turn>user\n{msg}<end_of_turn>\n
///   <start_of_turn>model\n{response}<end_of_turn>\n
///   ...
///   <start_of_turn>user\n{current}<end_of_turn>\n
///   <start_of_turn>model\n
fn build_conversation(history: &[Turn], current_user_msg: &str) -> String {
    let mut conv = String::new();
    for turn in history {
        conv.push_str("<start_of_turn>user\n");
        conv.push_str(&turn.user);
        conv.push_str("<end_of_turn>\n");
        conv.push_str("<start_of_turn>model\n");
        conv.push_str(&turn.assistant);
        conv.push_str("<end_of_turn>\n");
    }
    conv.push_str("<start_of_turn>user\n");
    conv.push_str(current_user_msg);
    conv.push_str("<end_of_turn>\n");
    conv.push_str("<start_of_turn>model\n");
    conv
}

/// Read a line from stdin, returning None on EOF.
fn read_input() -> Option<String> {
    use std::io::{self, BufRead, Write};
    print!("> ");
    io::stdout().flush().ok();
    let mut line = String::new();
    match io::stdin().lock().read_line(&mut line) {
        Ok(0) => None, // EOF
        Ok(_) => {
            let trimmed = line.trim_end().to_string();
            if trimmed.is_empty() {
                Some(trimmed)
            } else {
                Some(trimmed)
            }
        }
        Err(_) => None,
    }
}

fn run_chat_cpu(model: &Gemma, max_tokens: usize, sampling: SamplingOptions, stream: bool) {
    eprintln!("Chat mode (CPU). Type /quit to exit.\n");
    let mut history: Vec<Turn> = Vec::new();
    let mut cache_state = match model.init_cache_state() {
        Some(c) => c,
        None => {
            eprintln!("Full weights required for chat mode.");
            return;
        }
    };

    loop {
        let input = match read_input() {
            Some(s) if s == "/quit" || s == "/exit" => break,
            Some(s) if s.is_empty() => continue,
            Some(s) => s,
            None => break, // Ctrl+D
        };

        let chunk = build_conversation(&[], &input); // only the new turn
        let mut chunk_tokens = model.tokenizer.encode(&chunk);
        const BOS_ID: i32 = 2;
        if cache_state.pos == 0 {
            chunk_tokens.insert(0, BOS_ID);
        }
        let response = if stream {
            let mut stdout = std::io::stdout();
            let mut printed_any = false;
            let resp = model
                .generate_chat_streaming_with_cache(
                    &mut cache_state,
                    &chunk_tokens,
                    max_tokens,
                    sampling,
                    |piece| {
                        printed_any = true;
                        use std::io::Write;
                        print!("{}", piece);
                        stdout.flush().ok();
                    },
                )
                .unwrap_or_default();
            if printed_any {
                println!();
            }
            resp
        } else {
            let resp = model
                .generate_chat_with_cache(&mut cache_state, &chunk_tokens, max_tokens, sampling)
                .unwrap_or_default();
            println!("{}", resp);
            resp
        };

        history.push(Turn {
            user: input,
            assistant: response,
        });
    }
}

#[cfg(feature = "cuda")]
fn run_chat_cuda(
    model: &Gemma,
    max_tokens: usize,
    sampling: SamplingOptions,
    device_str: &str,
    stream: bool,
) {
    use gemma_gpu::backend::Backend;
    use gemma_gpu::cuda::CudaBackend;

    let ordinal = if device_str.starts_with("cuda:") {
        device_str[5..].parse::<usize>().unwrap_or(0)
    } else {
        0
    };

    let backend = match CudaBackend::new(ordinal) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to initialize CUDA device {ordinal}: {e}");
            std::process::exit(1);
        }
    };

    let caps = backend.caps();
    eprintln!(
        "Using GPU: {} (compute {}.{}, {}MB VRAM)",
        caps.name,
        caps.compute_major,
        caps.compute_minor,
        caps.total_memory / (1024 * 1024),
    );

    let mut gpu = match model.create_gpu_model(backend) {
        Some(Ok(g)) => g,
        Some(Err(e)) => {
            eprintln!("Failed to upload weights to GPU: {e}");
            std::process::exit(1);
        }
        None => {
            eprintln!("No full weights loaded, cannot use GPU");
            std::process::exit(1);
        }
    };

    eprintln!("Chat mode (CUDA). Type /quit to exit.\n");
    let mut history: Vec<Turn> = Vec::new();
    let mut cache_state = match model.init_cache_state() {
        Some(c) => c,
        None => {
            eprintln!("Full weights required for chat mode.");
            return;
        }
    };

    loop {
        let input = match read_input() {
            Some(s) if s == "/quit" || s == "/exit" => break,
            Some(s) if s.is_empty() => continue,
            Some(s) => s,
            None => break,
        };

        let chunk = build_conversation(&[], &input);
        let mut chunk_tokens = model.tokenizer.encode(&chunk);
        const BOS_ID: i32 = 2;
        if cache_state.pos == 0 {
            chunk_tokens.insert(0, BOS_ID);
        }
        let response = if stream {
            let mut stdout = std::io::stdout();
            let mut printed_any = false;
            let resp = model
                .generate_chat_gpu_streaming_with_cache(
                    &mut gpu,
                    &mut cache_state,
                    &chunk_tokens,
                    max_tokens,
                    sampling,
                    |piece| {
                        printed_any = true;
                        use std::io::Write;
                        print!("{}", piece);
                        stdout.flush().ok();
                    },
                )
                .unwrap_or_default();
            if printed_any {
                println!();
            }
            resp
        } else {
            let resp = model
                .generate_chat_gpu_with_cache(
                    &mut gpu,
                    &mut cache_state,
                    &chunk_tokens,
                    max_tokens,
                    sampling,
                )
                .unwrap_or_default();
            println!("{}", resp);
            resp
        };

        history.push(Turn {
            user: input,
            assistant: response,
        });
    }
}

// ── Single-shot CUDA ────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
fn run_cuda(
    model: &Gemma,
    prompt: &str,
    max_tokens: usize,
    sampling: SamplingOptions,
    device_str: &str,
) {
    use gemma_gpu::backend::Backend;
    use gemma_gpu::cuda::CudaBackend;

    let ordinal = if device_str.starts_with("cuda:") {
        device_str[5..].parse::<usize>().unwrap_or(0)
    } else {
        0
    };

    let backend = match CudaBackend::new(ordinal) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to initialize CUDA device {ordinal}: {e}");
            std::process::exit(1);
        }
    };

    let caps = backend.caps();
    eprintln!(
        "Using GPU: {} (compute {}.{}, {}MB VRAM)",
        caps.name,
        caps.compute_major,
        caps.compute_minor,
        caps.total_memory / (1024 * 1024),
    );

    let mut gpu = match model.create_gpu_model(backend) {
        Some(Ok(g)) => g,
        Some(Err(e)) => {
            eprintln!("Failed to upload weights to GPU: {e}");
            std::process::exit(1);
        }
        None => {
            eprintln!("No full weights loaded, cannot use GPU");
            std::process::exit(1);
        }
    };

    let out = model.generate_with_gpu(&mut gpu, prompt, max_tokens, sampling);
    println!("{}", out);
}

fn bench_series(mut run: impl FnMut() -> usize, warmup: usize, iters: usize) -> Vec<f64> {
    use std::time::Instant;
    for _ in 0..warmup {
        let _ = run();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        let tokens = run();
        let elapsed = start.elapsed().as_secs_f64();
        samples.push(tokens as f64 / elapsed.max(1e-9));
    }
    samples
}

fn summarize(samples: &[f64]) -> (f64, f64, f64) {
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = v[v.len() / 2];
    let p90_idx = ((v.len() as f64 * 0.9).floor() as usize).min(v.len() - 1);
    let p90 = v[p90_idx];
    let mean = v.iter().copied().sum::<f64>() / v.len() as f64;
    (mean, median, p90)
}

fn run_bench_cpu(
    model: &Gemma,
    prompt: &str,
    max_tokens: usize,
    sampling: SamplingOptions,
    warmup: usize,
    iters: usize,
    json_out: Option<&str>,
) {
    eprintln!("CPU bench: device=CPU");
    let tokenizer = &model.tokenizer;
    let samples = bench_series(
        || {
            let out = model.generate_with_sampling(prompt, max_tokens, sampling);
            tokenizer.encode(&out).len().max(1)
        },
        warmup,
        iters,
    );
    let (mean, med, p90) = summarize(&samples);
    eprintln!(
        "CPU bench: mean {:.2} tok/s, p50 {:.2}, p90 {:.2} (iters={}, warmup={})",
        mean, med, p90, iters, warmup
    );
    if let Some(path) = json_out {
        let payload = serde_json::json!({
            "device": "cpu",
            "mean_tok_s": mean,
            "p50_tok_s": med,
            "p90_tok_s": p90,
            "iters": iters,
            "warmup": warmup,
        });
        std::fs::write(path, payload.to_string()).ok();
    }
}

#[cfg(feature = "cuda")]
fn run_bench_cuda(
    model: &Gemma,
    prompt: &str,
    max_tokens: usize,
    sampling: SamplingOptions,
    device_str: &str,
    warmup: usize,
    iters: usize,
    json_out: Option<&str>,
) {
    use gemma_gpu::backend::Backend;
    use gemma_gpu::cuda::CudaBackend;

    let ordinal = if device_str.starts_with("cuda:") {
        device_str[5..].parse::<usize>().unwrap_or(0)
    } else {
        0
    };

    let backend = match CudaBackend::new(ordinal) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Failed to initialize CUDA device {ordinal}: {e}");
            return;
        }
    };
    let caps = backend.caps().clone();
    eprintln!(
        "CUDA bench: device={} compute {}.{} free={}MB total={}MB",
        caps.name,
        caps.compute_major,
        caps.compute_minor,
        caps.free_memory / (1024 * 1024),
        caps.total_memory / (1024 * 1024)
    );
    let mut gpu = match model.create_gpu_model(backend) {
        Some(Ok(g)) => g,
        Some(Err(e)) => {
            eprintln!("Failed to upload weights to GPU: {e}");
            return;
        }
        None => {
            eprintln!("No full weights loaded, cannot use GPU");
            return;
        }
    };

    let tokenizer = &model.tokenizer;
    let samples = bench_series(
        || {
            let out = model.generate_with_gpu(&mut gpu, prompt, max_tokens, sampling);
            tokenizer.encode(&out).len().max(1)
        },
        warmup,
        iters,
    );
    let (mean, med, p90) = summarize(&samples);
    eprintln!(
        "CUDA bench: mean {:.2} tok/s, p50 {:.2}, p90 {:.2} (iters={}, warmup={})",
        mean, med, p90, iters, warmup
    );
    if let Some(path) = json_out {
        let payload = serde_json::json!({
            "device": "cuda",
            "name": caps.name,
            "compute_major": caps.compute_major,
            "compute_minor": caps.compute_minor,
            "free_mb": caps.free_memory / (1024 * 1024),
            "total_mb": caps.total_memory / (1024 * 1024),
            "mean_tok_s": mean,
            "p50_tok_s": med,
            "p90_tok_s": p90,
            "iters": iters,
            "warmup": warmup,
        });
        std::fs::write(path, payload.to_string()).ok();
    }
}
