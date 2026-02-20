pub mod huggingface;
pub mod kaggle;

mod util;

use std::path::Path;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum FetchError {
    #[error("network error: {0}")]
    Network(String),
    #[error("auth error: {0}")]
    Auth(String),
    #[error("io error: {0}")]
    Io(String),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("not found: {0}")]
    NotFound(String),
}

pub type Result<T> = std::result::Result<T, FetchError>;

/// Download a file to disk, creating parent directories.
pub(crate) fn write_stream_to_path<R: std::io::Read, P: AsRef<Path>>(
    mut reader: R,
    dest: P,
) -> Result<()> {
    use std::fs::File;
    use std::io::copy;
    if let Some(parent) = dest.as_ref().parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| FetchError::Io(format!("create dir {}: {e}", parent.display())))?;
    }
    let mut f = File::create(&dest)
        .map_err(|e| FetchError::Io(format!("create {}: {e}", dest.as_ref().display())))?;
    copy(&mut reader, &mut f)
        .map_err(|e| FetchError::Io(format!("write {}: {e}", dest.as_ref().display())))?;
    Ok(())
}
