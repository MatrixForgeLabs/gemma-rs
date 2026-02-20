//! Kaggle Models API: search + download (blocking).

use std::path::{Path, PathBuf};

use reqwest::blocking::Client;
use reqwest::header::USER_AGENT;
use serde::Deserialize;
use url::Url;

use crate::{util, write_stream_to_path, FetchError, Result};

const KAGGLE_API: &str = "https://www.kaggle.com/api/v1";

#[derive(Debug, Clone)]
pub struct KaggleCreds {
    pub username: String,
    pub key: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KaggleModel {
    pub owner_slug: String,
    pub model_slug: String,
    pub framework: Option<String>,
    pub id: Option<String>,
    pub subtitle: Option<String>,
}

/// A fully-qualified Kaggle model handle: owner/model/framework/variation
#[derive(Debug, Clone)]
pub struct KaggleHandle {
    pub owner: String,
    pub model: String,
    pub framework: String,
    pub variation: String,
}

impl KaggleHandle {
    pub fn parse(spec: &str) -> Result<Self> {
        let parts: Vec<_> = spec.split('/').collect();
        if parts.len() != 4 {
            return Err(FetchError::Parse(
                "expected owner/model/framework/variation".into(),
            ));
        }
        Ok(Self {
            owner: parts[0].to_string(),
            model: parts[1].to_string(),
            framework: parts[2].to_string(),
            variation: parts[3].to_string(),
        })
    }

    pub fn to_path(&self) -> String {
        format!(
            "{}/{}/{}/{}",
            self.owner, self.model, self.framework, self.variation
        )
    }
}

fn load_creds() -> Result<KaggleCreds> {
    if let (Ok(u), Ok(k)) = (
        std::env::var("KAGGLE_USERNAME"),
        std::env::var("KAGGLE_KEY"),
    ) {
        return Ok(KaggleCreds {
            username: u,
            key: k,
        });
    }

    if let Some(path) = util::find_kaggle_json() {
        let content = util::read_to_string(&path)?;
        let v: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| FetchError::Parse(format!("kaggle.json: {e}")))?;
        let username = v
            .get("username")
            .and_then(|x| x.as_str())
            .ok_or_else(|| FetchError::Auth("kaggle.json missing username".into()))?;
        let key = v
            .get("key")
            .and_then(|x| x.as_str())
            .ok_or_else(|| FetchError::Auth("kaggle.json missing key".into()))?;
        return Ok(KaggleCreds {
            username: username.to_string(),
            key: key.to_string(),
        });
    }

    Err(FetchError::Auth(
        "set KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json".into(),
    ))
}

fn client() -> Result<Client> {
    Client::builder()
        .user_agent("gemma-rs-fetch/0.1")
        .build()
        .map_err(|e| FetchError::Network(format!("build client: {e}")))
}

/// Search Kaggle models (public).
pub fn search_models(query: &str, page_size: usize) -> Result<Vec<KaggleModel>> {
    let creds = load_creds()?;
    let url = Url::parse_with_params(
        &format!("{KAGGLE_API}/models/list"),
        &[("search", query), ("pageSize", &page_size.to_string())],
    )
    .map_err(|e| FetchError::Parse(format!("url: {e}")))?;

    let resp = client()?
        .get(url)
        .basic_auth(&creds.username, Some(&creds.key))
        .header(USER_AGENT, "gemma-rs-fetch/0.1")
        .send()
        .map_err(|e| FetchError::Network(format!("kaggle search: {e}")))?;

    if resp.status().is_success() {
        resp.json::<Vec<KaggleModel>>()
            .map_err(|e| FetchError::Parse(format!("kaggle search json: {e}")))
    } else {
        Err(FetchError::Network(format!(
            "kaggle search failed: {}",
            resp.status()
        )))
    }
}

/// Download the latest (or a specific) model version to dest_dir.
pub fn download_model(
    handle: &KaggleHandle,
    version: Option<u32>,
    dest_dir: &Path,
) -> Result<PathBuf> {
    let creds = load_creds()?;
    let base = if let Some(v) = version {
        format!(
            "{KAGGLE_API}/models/{}/{}/{}/{}/versions/{}/download",
            handle.owner, handle.model, handle.framework, handle.variation, v
        )
    } else {
        format!(
            "{KAGGLE_API}/models/{}/{}/{}/{}/download",
            handle.owner, handle.model, handle.framework, handle.variation
        )
    };
    let url = Url::parse(&base).map_err(|e| FetchError::Parse(format!("url: {e}")))?;

    let mut resp = client()?
        .get(url)
        .basic_auth(&creds.username, Some(&creds.key))
        .header(USER_AGENT, "gemma-rs-fetch/0.1")
        .send()
        .map_err(|e| FetchError::Network(format!("kaggle download: {e}")))?;

    let status = resp.status();
    if status == reqwest::StatusCode::UNAUTHORIZED {
        return Err(FetchError::Auth(
            "kaggle: check KAGGLE_USERNAME/KAGGLE_KEY or kaggle.json".into(),
        ));
    }
    if status == reqwest::StatusCode::NOT_FOUND {
        return Err(FetchError::NotFound(handle.to_path()));
    }
    if !status.is_success() {
        return Err(FetchError::Network(format!(
            "kaggle download failed: {}",
            status
        )));
    }

    // Pick filename from header or variation.
    let filename = resp
        .headers()
        .get(reqwest::header::CONTENT_DISPOSITION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| {
            s.split(';')
                .find_map(|part| part.trim().strip_prefix("filename="))
                .map(|f| f.trim_matches('"').to_string())
        })
        .unwrap_or_else(|| format!("{}-latest.bin", handle.variation));

    let dest = if dest_dir.is_dir() || dest_dir.extension().is_none() {
        dest_dir.join(filename)
    } else {
        dest_dir.to_path_buf()
    };

    resp = resp
        .error_for_status()
        .map_err(|e| FetchError::Network(format!("kaggle status: {e}")))?;

    write_stream_to_path(&mut resp, &dest)?;
    Ok(dest)
}
