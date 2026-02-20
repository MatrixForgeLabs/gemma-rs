use std::fs;

use gemma_io::blob_store::{BlobReader, BlobWriter};
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
fn blob_store_write_read_roundtrip() {
    let path = temp_path("blob_store");
    let pool = ThreadPool::new(2);

    let mut writer = BlobWriter::new(Path::new(path.path.clone()));
    writer.add("key1", b"hello", &pool);
    writer.add("key2", b"world!!!", &pool);
    writer.finalize();

    let reader = BlobReader::new(Path::new(path.path.clone()));
    assert_eq!(reader.keys().len(), 2);
    assert!(reader.find("key1").is_some());
    assert!(reader.find("key2").is_some());

    let mut got = Vec::new();
    let ok = reader.call_with_span::<u8, _>("key1", |span| {
        got.extend_from_slice(span);
    });
    assert!(ok);
    assert_eq!(got, b"hello");

    let mut got2 = Vec::new();
    let ok2 = reader.call_with_span::<u8, _>("key2", |span| {
        got2.extend_from_slice(span);
    });
    assert!(ok2);
    assert_eq!(got2, b"world!!!");

    let _ = fs::remove_file(&path.path);
}
