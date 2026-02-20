use std::fs;

use gemma_core::model_store::ModelStore;
use gemma_io::blob_store::BlobWriter;
use gemma_io::io::Path;
use gemma_threading::ThreadPool;

fn temp_path(name: &str) -> Path {
    let mut path = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    path.push(format!("gemma_rs_{}_{}_{}", name, pid, nanos));
    Path::new(path.to_string_lossy().to_string())
}

#[test]
fn model_store_reads_blob() {
    let path = temp_path("model_store");
    let pool = ThreadPool::new(2);
    let mut writer = BlobWriter::new(Path::new(path.path.clone()));
    writer.add("weights", b"abcd", &pool);
    writer.finalize();

    let store = ModelStore::new(Path::new(path.path.clone()));
    let data = store.read_blob("weights").unwrap();
    assert_eq!(data, b"abcd");

    let _ = fs::remove_file(&path.path);
}
