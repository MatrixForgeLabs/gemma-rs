//! Platform-independent file I/O.

use std::fs::{File as StdFile, OpenOptions};
#[cfg(not(any(unix, windows)))]
use std::io::{Read, Seek, SeekFrom, Write};

#[cfg(unix)]
use std::os::unix::fs::FileExt;

#[cfg(windows)]
use std::os::windows::fs::FileExt;

use memmap2::Mmap;

use crate::fields::SerializedSpan;

pub type MapPtr = Mmap;

pub trait File: Send + Sync {
    fn is_append_only(&self) -> bool;
    fn file_size(&self) -> u64;
    fn read(&self, offset: u64, size: u64, to: &mut [u8]) -> bool;
    fn write(&self, from: &[u8], size: u64, offset: u64) -> bool;
    fn map(&self) -> Option<MapPtr>;
    fn handle(&self) -> i32 {
        -1
    }
}

pub struct Path {
    pub path: String,
}

impl Path {
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }

    pub fn shortened(&self) -> String {
        const MAX_LEN: usize = 48;
        const CUT_POINT: usize = MAX_LEN / 2 - 5;
        if self.path.is_empty() {
            return "[no path specified]".to_string();
        }
        if self.path.len() > MAX_LEN {
            format!(
                "{} ... {}",
                &self.path[..CUT_POINT],
                &self.path[self.path.len() - CUT_POINT..]
            )
        } else {
            self.path.clone()
        }
    }

    pub fn exists(&self) -> bool {
        open_file_or_null(self, "r").is_some()
    }

    pub fn is_empty(&self) -> bool {
        self.path.is_empty()
    }
}

pub struct IOSpan {
    pub mem: *mut u8,
    pub bytes: usize,
}

pub struct IOBatch {
    offset: u64,
    total_bytes: u64,
    key_idx: usize,
    spans: Vec<IOSpan>,
}

impl IOBatch {
    pub fn new(offset: u64, key_idx: usize) -> Self {
        Self {
            offset,
            total_bytes: 0,
            key_idx,
            spans: Vec::new(),
        }
    }

    pub fn add(&mut self, mem: *mut u8, bytes: usize) -> bool {
        if self.spans.len() >= 1024 {
            return false;
        }
        if self.total_bytes + bytes as u64 > 0x7FFF_F000 {
            return false;
        }
        self.spans.push(IOSpan { mem, bytes });
        self.total_bytes += bytes as u64;
        true
    }

    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    pub fn key_idx(&self) -> usize {
        self.key_idx
    }

    pub fn read(&self, file: &dyn File) -> u64 {
        if self.spans.is_empty() {
            return 0;
        }
        let mut total = 0u64;
        let mut offset = self.offset;
        for span in &self.spans {
            let slice = unsafe { std::slice::from_raw_parts_mut(span.mem, span.bytes) };
            if !file.read(offset, span.bytes as u64, slice) {
                return 0;
            }
            total += span.bytes as u64;
            offset += span.bytes as u64;
        }
        total
    }
}

pub fn internal_init() {}

pub fn open_file_or_null(filename: &Path, mode: &str) -> Option<Box<dyn File>> {
    let is_read = !mode.starts_with('w');
    let mut options = OpenOptions::new();
    options.read(is_read);
    if !is_read {
        options.write(true).create(true).truncate(true);
    }

    let file = options.open(&filename.path).ok()?;
    Some(Box::new(FileStd { file }))
}

pub fn open_file_or_abort(filename: &Path, mode: &str) -> Box<dyn File> {
    open_file_or_null(filename, mode).unwrap_or_else(|| {
        panic!("Failed to open {}", filename.path);
    })
}

pub fn read_file_to_string(path: &Path) -> String {
    let file = open_file_or_abort(path, "r");
    let size = file.file_size();
    if size == 0 {
        panic!("Empty file {}", path.path);
    }
    let mut buf = vec![0u8; size as usize];
    if !file.read(0, size, &mut buf) {
        panic!("Failed to read {}", path.path);
    }
    String::from_utf8_lossy(&buf).to_string()
}

pub fn serialized_span_from_bytes(bytes: &[u8]) -> SerializedSpan<'_> {
    let len = bytes.len() / 4;
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u32, len) }
}

struct FileStd {
    file: StdFile,
}

impl File for FileStd {
    fn is_append_only(&self) -> bool {
        false
    }

    fn file_size(&self) -> u64 {
        self.file.metadata().map(|m| m.len()).unwrap_or(0)
    }

    fn read(&self, offset: u64, size: u64, to: &mut [u8]) -> bool {
        if to.len() < size as usize {
            return false;
        }
        #[cfg(any(unix, windows))]
        {
            let mut pos = 0usize;
            while pos < size as usize {
                match self
                    .file
                    .read_at(&mut to[pos..size as usize], offset + pos as u64)
                {
                    Ok(0) => break,
                    Ok(n) => pos += n,
                    Err(_) => return false,
                }
            }
            pos == size as usize
        }
        #[cfg(not(any(unix, windows)))]
        {
            let mut f = &self.file;
            if f.seek(SeekFrom::Start(offset)).is_err() {
                return false;
            }
            f.read_exact(&mut to[..size as usize]).is_ok()
        }
    }

    fn write(&self, from: &[u8], size: u64, offset: u64) -> bool {
        if from.len() < size as usize {
            return false;
        }
        #[cfg(any(unix, windows))]
        {
            let mut pos = 0usize;
            while pos < size as usize {
                match self
                    .file
                    .write_at(&from[pos..size as usize], offset + pos as u64)
                {
                    Ok(0) => break,
                    Ok(n) => pos += n,
                    Err(_) => return false,
                }
            }
            pos == size as usize
        }
        #[cfg(not(any(unix, windows)))]
        {
            let mut f = &self.file;
            if f.seek(SeekFrom::Start(offset)).is_err() {
                return false;
            }
            f.write_all(&from[..size as usize]).is_ok()
        }
    }

    fn map(&self) -> Option<MapPtr> {
        unsafe { Mmap::map(&self.file).ok() }
    }
}
