use gemma_core::configs::ModelConfig;
use gemma_core::gemma::{Gemma, SamplingOptions};
use serde::{Deserialize, Serialize};
use std::sync::mpsc as std_mpsc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use tokio::sync::{mpsc as tokio_mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;
use tower_http::cors::CorsLayer;

// ── OpenAI-compatible types ────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    #[allow(dead_code)]
    pub model: Option<String>,
    pub messages: Vec<Message>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub seed: Option<u64>,
}

fn default_max_tokens() -> usize {
    256
}

#[derive(Deserialize, Serialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: &'static str,
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Serialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: Delta,
    pub finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Serialize)]
pub struct ModelsResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

#[derive(Serialize)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: String,
}

// ── Message formatting ─────────────────────────────────────────────────

fn format_messages(messages: &[Message]) -> String {
    let mut prompt = String::new();
    let mut system_prefix = String::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                // Accumulate system messages to prepend to first user message
                if !system_prefix.is_empty() {
                    system_prefix.push('\n');
                }
                system_prefix.push_str(&msg.content);
            }
            "user" => {
                prompt.push_str("<start_of_turn>user\n");
                if !system_prefix.is_empty() {
                    prompt.push_str(&system_prefix);
                    prompt.push('\n');
                    system_prefix.clear();
                }
                prompt.push_str(&msg.content);
                prompt.push_str("<end_of_turn>\n");
            }
            "assistant" => {
                prompt.push_str("<start_of_turn>model\n");
                prompt.push_str(&msg.content);
                prompt.push_str("<end_of_turn>\n");
            }
            _ => {} // ignore unknown roles
        }
    }

    // Open the model's turn for completion
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

// ── Inference thread communication ─────────────────────────────────────

pub enum InferenceJob {
    Complete {
        prompt: String,
        max_tokens: usize,
        sampling: SamplingOptions,
        reply: oneshot::Sender<String>,
    },
    Stream {
        prompt: String,
        max_tokens: usize,
        sampling: SamplingOptions,
        chunks: tokio_mpsc::Sender<String>,
    },
}

#[derive(Clone)]
pub enum InferenceDevice {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(usize),
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Spawns a dedicated OS thread that owns the model.
/// Returns a channel sender for submitting jobs, after the model is loaded.
pub fn launch_inference_thread(
    weights_path: String,
    device: InferenceDevice,
) -> std_mpsc::SyncSender<InferenceJob> {
    let (job_tx, job_rx) = std_mpsc::sync_channel::<InferenceJob>(1);
    let (ready_tx, ready_rx) = std_mpsc::channel::<Result<(), String>>();

    std::thread::spawn(move || {
        // Load model inside this thread — Gemma is !Send, so it never leaves.
        let config = ModelConfig::new(0);
        let model = Gemma::from_sbs(config, &weights_path);

        if model.init_cache_state().is_none() {
            let _ = ready_tx.send(Err("No full weights found — cannot serve.".into()));
            return;
        }

        // Optionally set up GPU
        #[cfg(feature = "cuda")]
        let mut gpu_model = match &device {
            InferenceDevice::Cuda(ordinal) => {
                use gemma_gpu::backend::Backend;
                use gemma_gpu::cuda::CudaBackend;

                let backend = match CudaBackend::new(*ordinal) {
                    Ok(b) => b,
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!("CUDA init failed: {e}")));
                        return;
                    }
                };
                let caps = backend.caps();
                eprintln!(
                    "GPU: {} (compute {}.{}, {}MB VRAM)",
                    caps.name,
                    caps.compute_major,
                    caps.compute_minor,
                    caps.total_memory / (1024 * 1024),
                );
                match model.create_gpu_model(backend) {
                    Some(Ok(g)) => Some(g),
                    Some(Err(e)) => {
                        let _ = ready_tx.send(Err(format!("GPU weight upload failed: {e}")));
                        return;
                    }
                    None => {
                        let _ = ready_tx.send(Err("No full weights for GPU".into()));
                        return;
                    }
                }
            }
            InferenceDevice::Cpu => None,
        };

        let _ = ready_tx.send(Ok(()));

        // Process jobs forever
        while let Ok(job) = job_rx.recv() {
            match job {
                InferenceJob::Complete {
                    prompt,
                    max_tokens,
                    sampling,
                    reply,
                } => {
                    let mut cache = model.init_cache_state().unwrap();
                    let mut tokens = model.tokenizer.encode(&prompt);
                    const BOS_ID: i32 = 2;
                    tokens.insert(0, BOS_ID);

                    let result = run_inference(
                        &model,
                        &mut cache,
                        &tokens,
                        max_tokens,
                        sampling,
                        &device,
                        #[cfg(feature = "cuda")]
                        &mut gpu_model,
                    );

                    let _ = reply.send(result);
                }
                InferenceJob::Stream {
                    prompt,
                    max_tokens,
                    sampling,
                    chunks,
                } => {
                    let mut cache = model.init_cache_state().unwrap();
                    let mut tokens = model.tokenizer.encode(&prompt);
                    const BOS_ID: i32 = 2;
                    tokens.insert(0, BOS_ID);

                    run_inference_streaming(
                        &model,
                        &mut cache,
                        &tokens,
                        max_tokens,
                        sampling,
                        &device,
                        #[cfg(feature = "cuda")]
                        &mut gpu_model,
                        &chunks,
                    );
                    // Drop chunks sender — signals end to receiver
                }
            }
        }
    });

    // Wait for model to load
    match ready_rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            eprintln!("Inference thread failed: {e}");
            std::process::exit(1);
        }
        Err(_) => {
            eprintln!("Inference thread died during startup");
            std::process::exit(1);
        }
    }

    job_tx
}

/// Non-streaming inference (returns full text).
fn run_inference(
    model: &Gemma,
    cache: &mut gemma_core::gemma::CacheState,
    tokens: &[i32],
    max_tokens: usize,
    sampling: SamplingOptions,
    device: &InferenceDevice,
    #[cfg(feature = "cuda")] gpu_model: &mut Option<
        gemma_core::gpu_inference::GpuModel<gemma_gpu::cuda::CudaBackend>,
    >,
) -> String {
    match device {
        InferenceDevice::Cpu => model
            .generate_chat_with_cache(cache, tokens, max_tokens, sampling)
            .unwrap_or_default(),
        #[cfg(feature = "cuda")]
        InferenceDevice::Cuda(_) => {
            if let Some(gpu) = gpu_model.as_mut() {
                model
                    .generate_chat_gpu_with_cache(gpu, cache, tokens, max_tokens, sampling)
                    .unwrap_or_default()
            } else {
                model
                    .generate_chat_with_cache(cache, tokens, max_tokens, sampling)
                    .unwrap_or_default()
            }
        }
    }
}

/// Streaming inference — sends each text piece through the channel.
fn run_inference_streaming(
    model: &Gemma,
    cache: &mut gemma_core::gemma::CacheState,
    tokens: &[i32],
    max_tokens: usize,
    sampling: SamplingOptions,
    device: &InferenceDevice,
    #[cfg(feature = "cuda")] gpu_model: &mut Option<
        gemma_core::gpu_inference::GpuModel<gemma_gpu::cuda::CudaBackend>,
    >,
    chunks: &tokio_mpsc::Sender<String>,
) {
    let on_text = |piece: &str| {
        // blocking_send from sync thread into tokio channel
        let _ = chunks.blocking_send(piece.to_string());
    };

    match device {
        InferenceDevice::Cpu => {
            let _ = model.generate_chat_streaming_with_cache(
                cache, tokens, max_tokens, sampling, on_text,
            );
        }
        #[cfg(feature = "cuda")]
        InferenceDevice::Cuda(_) => {
            if let Some(gpu) = gpu_model.as_mut() {
                let _ = model.generate_chat_gpu_streaming_with_cache(
                    gpu, cache, tokens, max_tokens, sampling, on_text,
                );
            } else {
                let _ = model.generate_chat_streaming_with_cache(
                    cache, tokens, max_tokens, sampling, on_text,
                );
            }
        }
    }
}

// ── Axum handlers ──────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    pub job_tx: std_mpsc::SyncSender<InferenceJob>,
    pub model_name: String,
}

pub async fn handle_chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<serde_json::Value>)> {
    let prompt = format_messages(&req.messages);
    let temperature = req.temperature.unwrap_or(1.0);
    let top_k = if temperature == 0.0 { 1 } else { 40 };
    let top_p = req.top_p.unwrap_or(1.0);
    let sampling = SamplingOptions {
        temperature: if temperature == 0.0 {
            1.0
        } else {
            temperature
        },
        top_k,
        top_p,
        seed: req.seed,
    };

    let stream = req.stream.unwrap_or(false);
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
    let model_name = state.model_name.clone();

    if stream {
        // Streaming SSE response
        let (chunk_tx, chunk_rx) = tokio_mpsc::channel::<String>(64);

        let job = InferenceJob::Stream {
            prompt,
            max_tokens: req.max_tokens,
            sampling,
            chunks: chunk_tx,
        };

        if state.job_tx.try_send(job).is_err() {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": {
                        "message": "Server busy — try again shortly",
                        "type": "server_error",
                        "code": "capacity"
                    }
                })),
            ));
        }

        let created = unix_timestamp();
        let id_clone = id.clone();
        let model_clone = model_name.clone();

        // Build SSE stream
        let mut rx_stream = ReceiverStream::new(chunk_rx);

        let stream = async_stream::stream! {
            // First chunk: role announcement
            {
                let chunk = ChatCompletionChunk {
                    id: id_clone.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_clone.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: Some("assistant".into()),
                            content: None,
                        },
                        finish_reason: None,
                    }],
                };
                yield Ok::<_, std::convert::Infallible>(
                    Event::default().data(serde_json::to_string(&chunk).unwrap())
                );
            }

            // Content chunks
            while let Some(text) = rx_stream.next().await {
                let chunk = ChatCompletionChunk {
                    id: id_clone.clone(),
                    object: "chat.completion.chunk",
                    created,
                    model: model_clone.clone(),
                    choices: vec![ChunkChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: Some(text),
                        },
                        finish_reason: None,
                    }],
                };
                yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));
            }

            // Final chunk: finish_reason
            let chunk = ChatCompletionChunk {
                id: id_clone.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_clone.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some("stop"),
                }],
            };
            yield Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()));

            // [DONE] sentinel
            yield Ok(Event::default().data("[DONE]".to_string()));
        };

        Ok(Sse::new(stream).into_response())
    } else {
        // Non-streaming response
        let (reply_tx, reply_rx) = oneshot::channel();

        let job = InferenceJob::Complete {
            prompt: prompt.clone(),
            max_tokens: req.max_tokens,
            sampling,
            reply: reply_tx,
        };

        if state.job_tx.try_send(job).is_err() {
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": {
                        "message": "Server busy — try again shortly",
                        "type": "server_error",
                        "code": "capacity"
                    }
                })),
            ));
        }

        let content = reply_rx.await.unwrap_or_default();
        let created = unix_timestamp();

        let response = ChatCompletionResponse {
            id,
            object: "chat.completion",
            created,
            model: model_name,
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: "assistant".into(),
                    content: content.clone(),
                },
                finish_reason: "stop",
            }],
            usage: Usage {
                prompt_tokens: 0, // We don't track this precisely
                completion_tokens: 0,
                total_tokens: 0,
            },
        };

        Ok(Json(response).into_response())
    }
}

pub async fn handle_models(State(state): State<AppState>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list",
        data: vec![ModelObject {
            id: state.model_name.clone(),
            object: "model",
            created: 0,
            owned_by: "gemma-rs".into(),
        }],
    })
}

// ── Server entry point ─────────────────────────────────────────────────

pub fn run_server(
    weights_path: String,
    device: InferenceDevice,
    bind_addr: String,
    model_name: String,
) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to build tokio runtime");

    rt.block_on(async move {
        eprintln!("Loading model from {weights_path}...");
        let job_tx = launch_inference_thread(weights_path, device);
        eprintln!("Model loaded. Starting server on {bind_addr}");

        let state = AppState {
            job_tx,
            model_name,
        };

        let app = Router::new()
            .route("/v1/chat/completions", post(handle_chat_completions))
            .route("/v1/models", get(handle_models))
            .layer(CorsLayer::permissive())
            .with_state(state);

        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .unwrap_or_else(|e| {
                eprintln!("Failed to bind {bind_addr}: {e}");
                std::process::exit(1);
            });

        eprintln!("Listening on http://{bind_addr}");
        eprintln!("  POST /v1/chat/completions");
        eprintln!("  GET  /v1/models");

        axum::serve(listener, app)
            .await
            .expect("Server error");
    });
}
