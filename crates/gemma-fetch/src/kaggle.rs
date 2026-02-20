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
    Bearer { token: String },
    Basic { username: String, key: String },
}

// Kaggle token format used by the Models API:
// Authorization: KaggleToken <token>
const KAGGLE_SCHEME: &str = "KaggleToken";
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

    // Prefer basic creds if present (kaggle.json or env), fall back to API token.
    // This matches the behavior of Kaggle UI download snippets.

    // Look for kaggle.json in cwd.
    let kaggle_json_path = std::path::Path::new("kaggle.json");
    if kaggle_json_path.exists() {
        if let Ok(content) = util::read_to_string(&kaggle_json_path.to_path_buf()) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&content) {
                if let (Some(u), Some(k)) = (
                    v.get("username").and_then(|x| x.as_str()),
                    v.get("key").and_then(|x| x.as_str()),
                ) {
                    return Ok(KaggleAuth::Basic {
                        username: u.to_string(),
                        key: k.to_string(),
                    });
                }
            }
        }
    }

    if let (Some(u), Some(k)) = (env_var("KAGGLE_USERNAME"), env_var("KAGGLE_KEY")) {
        return Ok(KaggleAuth::Basic { username: u, key: k });
    }

    if let Some(tok) = env_var("KAGGLE_API_TOKEN").or_else(|| env_var("KAGGLE_TOKEN")) {
        return Ok(KaggleAuth::Bearer { token: tok });
    }

    Err(FetchError::Auth(
        "provide kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY, or set KAGGLE_API_TOKEN".into(),
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
    let mut version = version_override.or(handle.version);
    // If no version specified, attempt to discover latest version.
    if version.is_none() {
        if let Some(v) = latest_version(&auth, handle)? {
            version = Some(v);
        }
    }
    let base = if let Some(v) = version {
        format!(
            "{KAGGLE_API}/models/{}/{}/{}/{}/{v}/download",
            handle.owner, handle.model, handle.framework, handle.variation
        )
    } else {
        format!(
            "{KAGGLE_API}/models/{}/{}/{}/{}/download",
            handle.owner, handle.model, handle.framework, handle.variation
        )
    };
    let url = Url::parse(&base).map_err(|e| FetchError::Parse(format!("url: {e}")))?;

    eprintln!("Kaggle download: {}", url);

    let req = client()?
        .get(url)
        .header(USER_AGENT, "kagglehub/0.4.1")
        .header(reqwest::header::ACCEPT, "application/octet-stream");

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

    // Kaggle wraps model downloads in tar.gz archives.
    // Download to a temp file first, then extract if it's an archive.
    let tmp = dest.with_extension("download.tmp");
    write_stream_to_path(&mut resp, &tmp)?;

    match try_extract_tar_gz(&tmp, dest_dir) {
        Ok(extracted) => {
            let _ = std::fs::remove_file(&tmp);
            eprintln!("Extracted: {}", extracted.display());
            Ok(extracted)
        }
        Err(_) => {
            // Not an archive — just rename the temp file.
            std::fs::rename(&tmp, &dest)
                .map_err(|e| FetchError::Io(format!("rename: {e}")))?;
            Ok(dest)
        }
    }
}

/// If `path` is a tar.gz, extract its contents to `dest_dir` and return the
/// path to the first extracted file.
fn try_extract_tar_gz(path: &Path, dest_dir: &Path) -> Result<PathBuf> {
    use flate2::read::GzDecoder;
    use std::fs::File;

    let f = File::open(path).map_err(|e| FetchError::Io(format!("open: {e}")))?;
    let gz = GzDecoder::new(f);
    let mut archive = tar::Archive::new(gz);

    let mut first_file: Option<PathBuf> = None;
    for entry in archive
        .entries()
        .map_err(|e| FetchError::Io(format!("tar entries: {e}")))?
    {
        let mut entry = entry.map_err(|e| FetchError::Io(format!("tar entry: {e}")))?;
        let entry_path = entry
            .path()
            .map_err(|e| FetchError::Io(format!("tar path: {e}")))?
            .into_owned();
        let out = dest_dir.join(&entry_path);
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| FetchError::Io(format!("mkdir: {e}")))?;
        }
        entry
            .unpack(&out)
            .map_err(|e| FetchError::Io(format!("unpack: {e}")))?;
        if first_file.is_none() && entry_path.extension().is_some() {
            first_file = Some(out);
        }
    }
    first_file.ok_or_else(|| FetchError::Io("empty archive".into()))
}

#[derive(serde::Deserialize)]
struct InstanceInfo {
    #[serde(rename = "versionNumber")]
    version_number: Option<u32>,
}

fn latest_version(auth: &KaggleAuth, handle: &KaggleHandle) -> Result<Option<u32>> {
    let url = format!(
        "{KAGGLE_API}/models/{}/{}/{}/{}/get",
        handle.owner, handle.model, handle.framework, handle.variation
    );
    let req = client()?.get(url).header(USER_AGENT, "gemma-rs-fetch/0.1");
    let resp = send_with_auth(req, auth)?;
    if resp.status().is_success() {
        let info: InstanceInfo = resp
            .json()
            .map_err(|e| FetchError::Parse(format!("instance json: {e}")))?;
        Ok(info.version_number)
    } else if resp.status() == reqwest::StatusCode::NOT_FOUND {
        Ok(None)
    } else {
        Err(FetchError::Network(format!(
            "instance query failed: {}",
            resp.status()
        )))
    }
}
fn send_with_auth(builder: reqwest::blocking::RequestBuilder, auth: &KaggleAuth) -> Result<reqwest::blocking::Response> {
    let builder = match auth {
        KaggleAuth::Bearer { token } => builder
            .bearer_auth(token)
            .header(AUTHORIZATION, format!("{KAGGLE_SCHEME} {token}"))
            .header(KAGGLE_HEADER, format!("{KAGGLE_SCHEME} {token}")),
        KaggleAuth::Basic { username, key } => builder.basic_auth(username, Some(key)),
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
                .header(AUTHORIZATION, match auth {
                    KaggleAuth::Bearer { token } => format!("Bearer {token}"),
                    _ => unreachable!(),
                })
                .send()
                .map_err(|e| FetchError::Network(format!("kaggle retry: {e}")))?;
            return Ok(retry_resp);
        }
    }
    Ok(resp)
}
