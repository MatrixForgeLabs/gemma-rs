use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

use directories::BaseDirs;

use crate::{FetchError, Result};

/// Read a text file into a string.
pub(crate) fn read_to_string(path: &PathBuf) -> Result<String> {
    let mut buf = String::new();
    File::open(path)
        .map_err(|e| FetchError::Io(format!("open {}: {e}", path.display())))?
        .read_to_string(&mut buf)
        .map_err(|e| FetchError::Io(format!("read {}: {e}", path.display())))?;
    Ok(buf)
}

/// Locate kaggle.json (same search order as kaggle CLI).
pub(crate) fn find_kaggle_json() -> Option<PathBuf> {
    // Standard location ~/.kaggle/kaggle.json
    if let Some(base) = BaseDirs::new() {
        let path = base.home_dir().join(".kaggle").join("kaggle.json");
        if path.exists() {
            return Some(path);
        }
    }
    None
}
