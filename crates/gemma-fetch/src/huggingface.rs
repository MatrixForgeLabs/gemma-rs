//! Minimal Hugging Face Hub search + download (blocking).

use std::path::{Path, PathBuf};

use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, USER_AGENT};
use serde::Deserialize;
use url::Url;

use crate::{write_stream_to_path, FetchError, Result};

const HF_API: &str = "https://huggingface.co/api";

#[derive(Debug, Clone, Deserialize)]
pub struct HfModelInfo {
    #[serde(rename = "modelId")]
    pub model_id: String,
    pub pipeline_tag: Option<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub tags: Option<Vec<String>>,
}

fn client(token: Option<&str>) -> Result<Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    headers.insert(USER_AGENT, format!("gemma-rs-fetch/0.1").parse().unwrap());
    if let Some(tok) = token {
        headers.insert(
            AUTHORIZATION,
            format!("Bearer {}", tok.trim()).parse().unwrap(),
        );
    }
    Client::builder()
        .default_headers(headers)
        .build()
        .map_err(|e| FetchError::Network(format!("build client: {e}")))
}

/// Search HF models by query string.
pub fn search_models(query: &str, limit: usize, token: Option<&str>) -> Result<Vec<HfModelInfo>> {
    let url = Url::parse_with_params(
        &format!("{HF_API}/models"),
        &[("search", query), ("limit", &limit.to_string())],
    )
    .map_err(|e| FetchError::Parse(format!("url: {e}")))?;

    let resp = client(token)?
        .get(url)
        .send()
        .map_err(|e| FetchError::Network(format!("hf search: {e}")))?;
    if resp.status().is_success() {
        resp.json::<Vec<HfModelInfo>>()
            .map_err(|e| FetchError::Parse(format!("hf search json: {e}")))
    } else {
        Err(FetchError::Network(format!(
            "hf search failed: {}",
            resp.status()
        )))
    }
}

/// Download a file from a repo to dest path. If `dest_dir` is a directory, the
/// filename is taken from `file`.
pub fn download_file(
    repo: &str,
    file: &str,
    revision: Option<&str>,
    token: Option<&str>,
    dest_dir: &Path,
) -> Result<PathBuf> {
    let rev = revision.unwrap_or("main");
    let url = format!("https://huggingface.co/{repo}/resolve/{rev}/{file}");

    let dest_path = if dest_dir.is_dir() || dest_dir.extension().is_none() {
        dest_dir.join(Path::new(file).file_name().unwrap())
    } else {
        dest_dir.to_path_buf()
    };

    let mut resp = client(token)?
        .get(url)
        .send()
        .map_err(|e| FetchError::Network(format!("hf download: {e}")))?;

    match resp.status().as_u16() {
        200 => {
            resp = resp
                .error_for_status()
                .map_err(|e| FetchError::Network(format!("hf status: {e}")))?;
            write_stream_to_path(&mut resp, &dest_path)?;
            Ok(dest_path)
        }
        401 | 403 => Err(FetchError::Auth(
            "huggingface: missing or invalid token".into(),
        )),
        404 => Err(FetchError::NotFound(format!(
            "huggingface file {repo}/{file}@{rev}"
        ))),
        code => Err(FetchError::Network(format!("hf download failed: {code}"))),
    }
}
