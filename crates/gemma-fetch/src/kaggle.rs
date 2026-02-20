//! Kaggle Models API: search + download (blocking).

use std::path::{Path, PathBuf};

use reqwest::blocking::Client;
use reqwest::header::{AUTHORIZATION, USER_AGENT};
use serde::Deserialize;
use url::Url;

use crate::{util, write_stream_to_path, FetchError, Result};

const KAGGLE_API: &str = "https://www.kaggle.com/api/v1";

#[derive(Debug, Clone)]
pub struct KaggleCreds {
    pub username: String,
    pub key: String,
}

#[derive(Debug, Clone)]
enum KaggleAuth {
    Basic { username: String, key: String },
    Bearer { token: String },
}

// Newer Kaggle "API Token" (from account page) is used as:
// Authorization: KaggleToken <token>
const KAGGLE_BEARER_SCHEME: &str = "KaggleToken";
const KAGGLE_HEADER: &str = "X-Kaggle-Authorization";

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
    pub version: Option<u32>,
}

impl KaggleHandle {
    pub fn parse(spec: &str) -> Result<Self> {
        let parts: Vec<_> = spec.split('/').collect();
        if !(4..=5).contains(&parts.len()) {
            return Err(FetchError::Parse(
                "expected owner/model/framework/variation[/version]".into(),
            ));
        }
        let version = if parts.len() == 5 {
            Some(
                parts[4]
                    .parse::<u32>()
                    .map_err(|_| FetchError::Parse("version must be an integer".into()))?,
            )
        } else {
            None
        };
        Ok(Self {
            owner: parts[0].to_string(),
            model: parts[1].to_string(),
            framework: parts[2].to_string(),
            variation: parts[3].to_string(),
            version,
        })
    }

    pub fn to_path(&self) -> String {
        format!(
            "{}/{}/{}/{}",
            self.owner, self.model, self.framework, self.variation
        )
    }
}

fn env_var(name: &str) -> Option<String> {
    std::env::var(name).ok().map(|v| v.trim().to_string()).filter(|s| !s.is_empty())
}

fn load_auth() -> Result<KaggleAuth> {
    // Best-effort .env load so callers don't have to do it explicitly.
    let _ = dotenvy::dotenv();

    if let Some(tok) = env_var("KAGGLE_API_TOKEN").or_else(|| env_var("KAGGLE_TOKEN")) {
        // Accept "username:key" or JSON {"username": "...", "key": "..."}
        if let Some((u, k)) = tok.split_once(':') {
            return Ok(KaggleAuth::Basic {
                username: u.to_string(),
                key: k.to_string(),
            });
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&tok) {
            let username = v
                .get("username")
                .and_then(|x| x.as_str())
                .ok_or_else(|| FetchError::Auth("KAGGLE_API_TOKEN missing username".into()))?;
            let key = v
                .get("key")
                .and_then(|x| x.as_str())
                .ok_or_else(|| FetchError::Auth("KAGGLE_API_TOKEN missing key".into()))?;
            return Ok(KaggleAuth::Basic {
                username: username.to_string(),
                key: key.to_string(),
            });
        }
        // If token is only the key, allow username from env.
        if let Ok(u) = std::env::var("KAGGLE_USERNAME") {
            return Ok(KaggleAuth::Basic {
                username: u,
                key: tok,
            });
        }
        // Otherwise treat as bearer PAT.
        return Ok(KaggleAuth::Bearer { token: tok });
    }

    if let (Ok(u), Ok(k)) = (
        std::env::var("KAGGLE_USERNAME"),
        std::env::var("KAGGLE_KEY"),
    ) {
        return Ok(KaggleAuth::Basic {
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
        return Ok(KaggleAuth::Basic {
            username: username.to_string(),
            key: key.to_string(),
        });
    }

    Err(FetchError::Auth(
        "set KAGGLE_API_TOKEN, or KAGGLE_USERNAME/KAGGLE_KEY, or ~/.kaggle/kaggle.json".into(),
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
    let auth = load_auth()?;
    let url = Url::parse_with_params(
        &format!("{KAGGLE_API}/models/list"),
        &[("search", query), ("pageSize", &page_size.to_string())],
    )
    .map_err(|e| FetchError::Parse(format!("url: {e}")))?;

    let req = client()?
        .get(url)
        .header(USER_AGENT, "gemma-rs-fetch/0.1");

    let resp = send_with_auth(req, &auth)?;

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
    version_override: Option<u32>,
    dest_dir: &Path,
) -> Result<PathBuf> {
    let auth = load_auth()?;
    let version = version_override.or(handle.version);
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

    let req = client()?
        .get(url)
        .header(USER_AGENT, "gemma-rs-fetch/0.1");

    let mut resp = send_with_auth(req, &auth)?;

    let status = resp.status();
    if status == reqwest::StatusCode::UNAUTHORIZED {
        return Err(FetchError::Auth(
            "kaggle: check KAGGLE_API_TOKEN (or KAGGLE_TOKEN), KAGGLE_USERNAME/KAGGLE_KEY, or kaggle.json".into(),
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
fn send_with_auth(builder: reqwest::blocking::RequestBuilder, auth: &KaggleAuth) -> Result<reqwest::blocking::Response> {
    let builder = match auth {
        KaggleAuth::Basic { username, key } => builder.basic_auth(username, Some(key)),
        KaggleAuth::Bearer { token } => builder
            .header(AUTHORIZATION, format!("{KAGGLE_BEARER_SCHEME} {token}"))
            .header(KAGGLE_HEADER, format!("{KAGGLE_BEARER_SCHEME} {token}")),
    };
    let retry_clone = if matches!(auth, KaggleAuth::Bearer { .. }) {
        builder.try_clone()
    } else {
        None
    };

    let resp = builder
        .send()
        .map_err(|e| FetchError::Network(format!("kaggle request: {e}")))?;

    // If bearer with KaggleToken fails 401, retry with plain Bearer to be compatible with older servers.
    if matches!(auth, KaggleAuth::Bearer { .. }) && resp.status() == reqwest::StatusCode::UNAUTHORIZED {
        if let Some(clone) = retry_clone {
            let retry_resp = clone
                .header(AUTHORIZATION, format!("Bearer {}", match auth { KaggleAuth::Bearer { token } => token, _ => unreachable!() }))
                .send()
                .map_err(|e| FetchError::Network(format!("kaggle retry: {e}")))?;
            return Ok(retry_resp);
        }
    }
    Ok(resp)
}
