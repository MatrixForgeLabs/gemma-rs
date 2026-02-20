//! Blob store reader/writer.

use std::collections::{HashMap, HashSet};

use crate::io::{open_file_or_abort, open_file_or_null, File, Path};
use gemma_threading::{parallel_for, ParallelismStrategy, ThreadPool};

#[derive(Copy, Clone, Debug, Default)]
pub struct BlobRange {
    pub offset: u64,
    pub bytes: usize,
    pub key_idx: usize,
}

impl BlobRange {
    pub fn end(&self) -> u64 {
        self.offset + self.bytes as u64
    }
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct Header {
    magic: u32,
    num_blobs: u32,
    file_bytes: u64,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
struct U128 {
    lo: u64,
    hi: u64,
}

impl U128 {
    fn from_bytes(bytes: [u8; 16]) -> Self {
        let lo = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let hi = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        Self { lo, hi }
    }

    fn to_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[0..8].copy_from_slice(&self.lo.to_le_bytes());
        out[8..16].copy_from_slice(&self.hi.to_le_bytes());
        out
    }
}

const K_MAGIC: u32 = 0x0A53_4253; // SBS\n
const K_BLOB_ALIGN: usize = 256;
const K_END_ALIGN: usize = 64 * 1024;
const K_U128_BYTES: usize = 16;
const K_MAX_BLOBS: usize = 16 * 1024;

fn round_up_to_align(size_or_offset: usize) -> usize {
    let align = K_BLOB_ALIGN;
    ((size_or_offset + align - 1) / align) * align
}

fn padded_header_and_dir_bytes(num_blobs: usize) -> usize {
    assert!(num_blobs < K_MAX_BLOBS);
    round_up_to_align(core::mem::size_of::<Header>() + 2 * K_U128_BYTES * num_blobs)
}

fn padded_payload_bytes(blob_sizes: &[usize]) -> u64 {
    let mut total = 0u64;
    for &blob in blob_sizes {
        total += round_up_to_align(blob) as u64;
    }
    total
}

fn key_from_string(key: &str) -> U128 {
    let bytes = key.as_bytes();
    if bytes.is_empty() || bytes.len() > K_U128_BYTES {
        panic!("Key {} is too long, please truncate to 16 chars.", key);
    }
    let mut buf = [0u8; 16];
    buf[..bytes.len()].copy_from_slice(bytes);
    U128::from_bytes(buf)
}

fn string_from_key(key: U128) -> String {
    let bytes = key.to_bytes();
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).to_string()
}

struct BlobStore {
    is_file_v2: bool,
    header: Header,
    directory: Vec<U128>,
}

impl BlobStore {
    fn parse_header_and_directory_v1(file: &dyn File) -> Option<Self> {
        let mut header = Header::default();
        if !file.read(
            0,
            core::mem::size_of::<Header>() as u64,
            header_as_mut_bytes(&mut header),
        ) {
            return None;
        }
        if header.magic != K_MAGIC
            || header.num_blobs == 0
            || header.num_blobs as usize > K_MAX_BLOBS
        {
            return None;
        }

        let num_blobs = header.num_blobs as usize;
        let directory_bytes = 2 * K_U128_BYTES * num_blobs;
        let mut directory = vec![U128::default(); num_blobs * 2];
        if !file.read(
            core::mem::size_of::<Header>() as u64,
            directory_bytes as u64,
            u128_vec_as_mut_bytes(&mut directory),
        ) {
            return None;
        }

        let bs = Self {
            is_file_v2: false,
            header,
            directory,
        };
        if !bs.is_valid(file.file_size()) {
            return None;
        }
        Some(bs)
    }

    fn parse_header_and_directory_v2(file: &dyn File) -> Option<Self> {
        let file_size = file.file_size();
        if file_size < core::mem::size_of::<Header>() as u64 {
            return None;
        }
        let mut header = Header::default();
        let header_offset = file_size - core::mem::size_of::<Header>() as u64;
        if !file.read(
            header_offset,
            core::mem::size_of::<Header>() as u64,
            header_as_mut_bytes(&mut header),
        ) {
            return None;
        }
        if header.magic != K_MAGIC || header.num_blobs as usize > K_MAX_BLOBS {
            return None;
        }

        let num_blobs = header.num_blobs as usize;
        let directory_bytes = 2 * K_U128_BYTES * num_blobs;
        let directory_offset = header_offset - directory_bytes as u64;
        let mut directory = vec![U128::default(); num_blobs * 2];
        if !file.read(
            directory_offset,
            directory_bytes as u64,
            u128_vec_as_mut_bytes(&mut directory),
        ) {
            return None;
        }

        let bs = Self {
            is_file_v2: true,
            header,
            directory,
        };
        if !bs.is_valid(file.file_size()) {
            return None;
        }
        Some(bs)
    }

    fn from_file(file: &dyn File) -> Self {
        if let Some(bs) = Self::parse_header_and_directory_v1(file) {
            return bs;
        }
        if let Some(bs) = Self::parse_header_and_directory_v2(file) {
            return bs;
        }
        panic!("Failed to read BlobStore header or directory.");
    }

    fn new(keys: &[U128], blob_sizes: &[usize]) -> Self {
        assert_eq!(keys.len(), blob_sizes.len());
        assert!(keys.len() < K_MAX_BLOBS);

        ensure_unique(keys);

        let num_blobs = keys.len();
        let mut header = Header {
            magic: K_MAGIC,
            num_blobs: num_blobs as u32,
            file_bytes: 0,
        };
        let size_before_blobs = bytes_before_blobs_v2().len();
        header.file_bytes = round_up_to_end_align(
            size_before_blobs as u64
                + padded_payload_bytes(blob_sizes)
                + padded_header_and_dir_bytes(num_blobs) as u64,
        );

        let mut directory = vec![U128::default(); num_blobs * 2];
        directory[..num_blobs].copy_from_slice(keys);

        let mut offset = size_before_blobs as u64;
        for i in 0..num_blobs {
            set_range(&mut directory, num_blobs, i, offset, blob_sizes[i]);
            offset = round_up_to_align((offset + blob_sizes[i] as u64) as usize) as u64;
        }

        let bs = Self {
            is_file_v2: true,
            header,
            directory,
        };
        assert!(bs.is_valid(bs.file_size()));
        bs
    }

    fn file_size(&self) -> u64 {
        self.header.file_bytes
    }

    fn num_blobs(&self) -> usize {
        self.header.num_blobs as usize
    }

    fn keys(&self) -> &[U128] {
        &self.directory[..self.num_blobs()]
    }

    fn get_range(&self, key_idx: usize) -> (u64, usize) {
        let val = self.directory[self.num_blobs() + key_idx];
        let offset = val.lo;
        let bytes = val.hi as usize;
        (offset, bytes)
    }

    fn is_valid(&self, file_size: u64) -> bool {
        if self.directory.is_empty() {
            return false;
        }
        if self.header.magic != K_MAGIC
            || self.header.num_blobs == 0
            || self.header.file_bytes != file_size
        {
            return false;
        }

        let size_before = self.bytes_before_blobs().len();
        let size_after = self.bytes_after_blobs().len();

        let mut expected_offset = size_before as u64;
        for key_idx in 0..self.num_blobs() {
            let (actual_offset, bytes) = self.get_range(key_idx);
            if expected_offset != actual_offset {
                return false;
            }
            expected_offset = round_up_to_align((expected_offset + bytes as u64) as usize) as u64;
        }

        if expected_offset != self.header.file_bytes
            && expected_offset + size_after as u64 != self.header.file_bytes
        {
            return false;
        }
        true
    }

    fn bytes_before_blobs(&self) -> Vec<u8> {
        if self.is_file_v2 {
            bytes_before_blobs_v2()
        } else {
            let padded = padded_header_and_dir_bytes(self.num_blobs());
            let mut buf = vec![0u8; padded];
            buf[..core::mem::size_of::<Header>()].copy_from_slice(header_as_bytes(&self.header));
            let dir_bytes = u128_vec_as_bytes(&self.directory);
            let offset = core::mem::size_of::<Header>();
            buf[offset..offset + dir_bytes.len()].copy_from_slice(dir_bytes);
            buf
        }
    }

    fn bytes_after_blobs(&self) -> Vec<u8> {
        let (last_offset, last_bytes) = self.get_range(self.num_blobs() - 1);
        let blob_end = round_up_to_align((last_offset + last_bytes as u64) as usize) as u64;

        if !self.is_file_v2 {
            let size = (self.file_size() - blob_end) as usize;
            return vec![0u8; size];
        }

        let size = (self.file_size() - blob_end) as usize;
        let mut buf = vec![0u8; size];
        let header_size = core::mem::size_of::<Header>();
        let directory_size = 2 * K_U128_BYTES * self.num_blobs();

        let mut offset = size - header_size;
        buf[offset..offset + header_size].copy_from_slice(header_as_bytes(&self.header));

        offset -= directory_size;
        buf[offset..offset + directory_size].copy_from_slice(u128_vec_as_bytes(&self.directory));

        buf
    }
}

fn bytes_before_blobs_v2() -> Vec<u8> {
    let header = Header {
        magic: K_MAGIC,
        num_blobs: 0,
        file_bytes: K_END_ALIGN as u64,
    };
    let mut buf = vec![0u8; padded_header_and_dir_bytes(0)];
    buf[..core::mem::size_of::<Header>()].copy_from_slice(header_as_bytes(&header));
    buf
}

fn round_up_to_end_align(value: u64) -> u64 {
    let align = K_END_ALIGN as u64;
    ((value + align - 1) / align) * align
}

fn set_range(directory: &mut [U128], num_blobs: usize, key_idx: usize, offset: u64, bytes: usize) {
    let val = U128 {
        lo: offset,
        hi: bytes as u64,
    };
    directory[num_blobs + key_idx] = val;
}

fn ensure_unique(keys: &[U128]) {
    let mut set = HashSet::new();
    for &key in keys {
        let s = string_from_key(key);
        if !set.insert(s) {
            panic!("Duplicate key");
        }
    }
}

fn header_as_bytes(header: &Header) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            header as *const Header as *const u8,
            core::mem::size_of::<Header>(),
        )
    }
}

fn header_as_mut_bytes(header: &mut Header) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(
            header as *mut Header as *mut u8,
            core::mem::size_of::<Header>(),
        )
    }
}

fn u128_vec_as_bytes(vec: &[U128]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(vec.as_ptr() as *const u8, vec.len() * K_U128_BYTES) }
}

fn u128_vec_as_mut_bytes(vec: &mut [U128]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(vec.as_mut_ptr() as *mut u8, vec.len() * K_U128_BYTES) }
}

pub struct BlobReader {
    blob_path: Path,
    file: Box<dyn File>,
    file_bytes: u64,
    keys: Vec<String>,
    ranges: Vec<BlobRange>,
    key_idx_for_key: HashMap<String, usize>,
}

impl BlobReader {
    pub fn new(blob_path: Path) -> Self {
        let file = open_file_or_abort(&blob_path, "r");
        let file_bytes = file.file_size();
        if file_bytes == 0 {
            panic!("Zero-sized file {}", blob_path.path);
        }

        let bs = BlobStore::from_file(file.as_ref());
        let mut keys = Vec::with_capacity(bs.num_blobs());
        for &key in bs.keys() {
            keys.push(string_from_key(key));
        }

        let mut ranges = Vec::with_capacity(bs.num_blobs());
        let mut key_idx_for_key = HashMap::new();
        for key_idx in 0..keys.len() {
            let (offset, bytes) = bs.get_range(key_idx);
            ranges.push(BlobRange {
                offset,
                bytes,
                key_idx,
            });
            key_idx_for_key.insert(keys[key_idx].clone(), key_idx);
        }

        Self {
            blob_path,
            file,
            file_bytes,
            keys,
            ranges,
            key_idx_for_key,
        }
    }

    pub fn blob_path(&self) -> &Path {
        &self.blob_path
    }

    pub fn file(&self) -> &dyn File {
        self.file.as_ref()
    }

    pub fn file_bytes(&self) -> u64 {
        self.file_bytes
    }

    pub fn map(&self) -> Option<crate::io::MapPtr> {
        self.file.map()
    }

    pub fn close_file(&mut self) {
        self.file = Box::new(DummyFile {});
    }

    pub fn keys(&self) -> &[String] {
        &self.keys
    }

    pub fn range(&self, key_idx: usize) -> &BlobRange {
        &self.ranges[key_idx]
    }

    pub fn find(&self, key: &str) -> Option<&BlobRange> {
        let idx = *self.key_idx_for_key.get(key)?;
        Some(&self.ranges[idx])
    }

    pub fn call_with_span<T: Copy, F: FnOnce(&[T])>(&self, key: &str, func: F) -> bool {
        let range = match self.find(key) {
            Some(r) => r,
            None => return false,
        };
        if range.bytes % core::mem::size_of::<T>() != 0 {
            return false;
        }
        let count = range.bytes / core::mem::size_of::<T>();
        let mut storage: Vec<T> = vec![unsafe { core::mem::zeroed() }; count];
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(storage.as_mut_ptr() as *mut u8, range.bytes) };
        if !self.file.read(range.offset, range.bytes as u64, bytes) {
            return false;
        }
        func(&storage);
        true
    }
}

struct DummyFile {}

impl File for DummyFile {
    fn is_append_only(&self) -> bool {
        true
    }
    fn file_size(&self) -> u64 {
        0
    }
    fn read(&self, _offset: u64, _size: u64, _to: &mut [u8]) -> bool {
        false
    }
    fn write(&self, _from: &[u8], _size: u64, _offset: u64) -> bool {
        false
    }
    fn map(&self) -> Option<crate::io::MapPtr> {
        None
    }
}

struct BlobIO {
    range: BlobRange,
    data: usize,
}

fn enqueue_chunks(
    key_idx: usize,
    mut offset: u64,
    bytes: u64,
    mut data: *const u8,
    writes: &mut Vec<BlobIO>,
) {
    const CHUNK_BYTES: u64 = 10 * 1024 * 1024;
    let end = offset + bytes;
    if end >= CHUNK_BYTES {
        while offset <= end - CHUNK_BYTES {
            writes.push(BlobIO {
                range: BlobRange {
                    offset,
                    bytes: CHUNK_BYTES as usize,
                    key_idx,
                },
                data: data as usize,
            });
            offset += CHUNK_BYTES;
            unsafe {
                data = data.add(CHUNK_BYTES as usize);
            }
        }
    }
    if offset != end {
        writes.push(BlobIO {
            range: BlobRange {
                offset,
                bytes: (end - offset) as usize,
                key_idx,
            },
            data: data as usize,
        });
    }

    let padding = round_up_to_align(bytes as usize) - bytes as usize;
    if padding > 0 {
        static ZEROS: [u8; K_BLOB_ALIGN] = [0u8; K_BLOB_ALIGN];
        writes.push(BlobIO {
            range: BlobRange {
                offset: end,
                bytes: padding,
                key_idx,
            },
            data: ZEROS.as_ptr() as usize,
        });
    }
}

pub struct BlobWriter {
    file: Box<dyn File>,
    keys: Vec<U128>,
    blob_sizes: Vec<usize>,
    curr_offset: u64,
}

impl BlobWriter {
    pub fn new(filename: Path) -> Self {
        let file = open_file_or_null(&filename, "w+").unwrap_or_else(|| {
            panic!("Failed to open for writing {}", filename.path);
        });
        let header_bytes = bytes_before_blobs_v2();
        file.write(&header_bytes, header_bytes.len() as u64, 0);
        let curr_offset = header_bytes.len() as u64;
        Self {
            file,
            keys: Vec::new(),
            blob_sizes: Vec::new(),
            curr_offset,
        }
    }

    pub fn add(&mut self, key: &str, data: &[u8], pool: &ThreadPool) {
        if data.is_empty() {
            panic!("Blob size must be non-zero");
        }
        let key_u128 = key_from_string(key);
        self.keys.push(key_u128);
        self.blob_sizes.push(data.len());

        let mut writes = Vec::new();
        enqueue_chunks(
            self.keys.len() - 1,
            self.curr_offset,
            data.len() as u64,
            data.as_ptr(),
            &mut writes,
        );

        let strategy = if self.file.is_append_only() {
            ParallelismStrategy::None
        } else {
            ParallelismStrategy::Flat
        };

        parallel_for(strategy, writes.len(), pool, |i, _worker| {
            let write = &writes[i];
            let ptr = write.data as *const u8;
            let bytes = unsafe { std::slice::from_raw_parts(ptr, write.range.bytes) };
            if !self
                .file
                .write(bytes, write.range.bytes as u64, write.range.offset)
            {
                panic!(
                    "Write failed for {}",
                    string_from_key(self.keys[write.range.key_idx])
                );
            }
        });

        self.curr_offset = writes.last().unwrap().range.end();
    }

    pub fn finalize(self) {
        if !self.file.is_append_only() && self.curr_offset != self.file.file_size() {
            // Leave a warning-like path as in C++.
        }
        let bs = BlobStore::new(&self.keys, &self.blob_sizes);
        let bytes_after = bs.bytes_after_blobs();
        self.file
            .write(&bytes_after, bytes_after.len() as u64, self.curr_offset);
    }
}
