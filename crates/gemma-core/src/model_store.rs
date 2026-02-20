//! Model store wrapper with MatPtr metadata support.

use std::collections::HashMap;

use crate::configs::ModelConfig;
use gemma_compression::types::Type;
use gemma_io::blob_store::{BlobRange, BlobReader};
use gemma_io::fields::read_fields;
use gemma_io::io::Path;
use gemma_util::basics::Extents2D;
use gemma_util::mat::MatPtr;

pub struct ModelStore {
    reader: BlobReader,
    mat_ptrs: Vec<MatPtr>,
    key_idx: Vec<usize>,
    mat_idx_for_name: HashMap<String, usize>,
    config: Option<ModelConfig>,
}

impl ModelStore {
    pub fn new(path: Path) -> Self {
        let reader = BlobReader::new(path);
        let mut store = Self {
            reader,
            mat_ptrs: Vec::new(),
            key_idx: Vec::new(),
            mat_idx_for_name: HashMap::new(),
            config: None,
        };
        store.read_config();
        store.read_mat_ptrs();
        store
    }

    pub fn read_blob(&self, key: &str) -> Option<Vec<u8>> {
        let range = self.reader.find(key)?;
        let mut buf = vec![0u8; range.bytes];
        if !self
            .reader
            .file()
            .read(range.offset, range.bytes as u64, &mut buf)
        {
            return None;
        }
        Some(buf)
    }

    pub fn read_blob_into(&self, key: &str, dst: &mut [u8]) -> bool {
        let range = match self.reader.find(key) {
            Some(r) => r,
            None => return false,
        };
        self.read_range_into(range, dst)
    }

    pub fn read_blob_string(&self, key: &str) -> Option<String> {
        let data = self.read_blob(key)?;
        Some(String::from_utf8_lossy(&data).to_string())
    }

    pub fn config(&self) -> Option<&ModelConfig> {
        self.config.as_ref()
    }

    pub fn mat_ptrs(&self) -> &[MatPtr] {
        &self.mat_ptrs
    }

    pub fn read_range_into(&self, range: &BlobRange, dst: &mut [u8]) -> bool {
        if dst.len() < range.bytes {
            return false;
        }
        self.reader
            .file()
            .read(range.offset, range.bytes as u64, &mut dst[..range.bytes])
    }

    pub fn find_and_update_mat_ptr(&self, mat: &mut MatPtr, preferred: Type) -> Option<BlobRange> {
        if !self.mat_ptrs.is_empty() {
            let mat_idx = *self.mat_idx_for_name.get(mat.name())?;
            let file_mat = &self.mat_ptrs[mat_idx];
            if file_mat.rows() != mat.rows() || file_mat.cols() != mat.cols() {
                panic!(
                    "Tensor {} shape {} {} mismatches file {} {}.",
                    mat.name(),
                    mat.rows(),
                    mat.cols(),
                    file_mat.rows(),
                    file_mat.cols()
                );
            }
            mat.set_type(file_mat.ty());
            mat.set_scale(file_mat.scale());
            let key_idx = self.key_idx[mat_idx];
            return Some(*self.reader.range(key_idx));
        }

        if let Some(range) = self.reader.find(mat.name()) {
            return Some(*range);
        }

        let candidates = if preferred != Type::Unknown {
            vec![preferred]
        } else {
            vec![Type::I8, Type::NUQ, Type::SFP, Type::BF16, Type::F32]
        };
        for ty in candidates {
            if let Some(range) = self
                .reader
                .find(&format!("{}{}", type_prefix(ty), mat.name()))
            {
                mat.set_type(ty);
                mat.set_scale(1.0);
                return Some(*range);
            }
        }
        None
    }

    pub fn mat_ptr(&self, name: &str) -> Option<MatPtr> {
        let mat_idx = *self.mat_idx_for_name.get(name)?;
        Some(self.mat_ptrs[mat_idx].clone())
    }

    fn read_mat_ptrs(&mut self) {
        if !self.reader.find("toc").is_some() {
            return;
        }

        let mut parsed = Vec::new();
        let ok = self.reader.call_with_span::<u32, _>("toc", |serialized| {
            let mut pos = 0usize;
            while pos < serialized.len() {
                let mut mat = MatPtr::new("", Type::Unknown, Extents2D::new(0, 0));
                let result = read_fields(&mut mat, serialized, pos);
                if result.pos == 0 {
                    panic!("Deserializing MatPtr {} failed", mat.name());
                }
                pos = result.pos + result.extra_u32 as usize;
                parsed.push(mat);
            }
        });
        if !ok {
            panic!("Failed to read toc MatPtrs");
        }

        for mat in parsed {
            let range = self.reader.find(mat.name()).unwrap_or_else(|| {
                panic!("MatPtr {} missing blob", mat.name());
            });
            let key_idx = range.key_idx;
            self.add_mat_ptr(key_idx, mat);
        }
    }

    fn add_mat_ptr(&mut self, key_idx: usize, mat: MatPtr) {
        let idx = self.mat_ptrs.len();
        self.mat_ptrs.push(mat);
        self.key_idx.push(key_idx);
        self.mat_idx_for_name
            .insert(self.mat_ptrs[idx].name().to_string(), idx);
    }

    fn read_config(&mut self) {
        if !self.reader.find("config").is_some() {
            return;
        }
        let mut config = ModelConfig::default();
        let ok = self
            .reader
            .call_with_span::<u32, _>("config", |serialized| {
                let result = read_fields(&mut config, serialized, 0);
                if result.pos == 0 {
                    panic!("Failed to read ModelConfig");
                }
            });
        if ok {
            self.config = Some(config);
        }
    }
}

fn type_prefix(ty: Type) -> char {
    match ty {
        Type::F32 => 'F',
        Type::BF16 => 'B',
        Type::SFP => '$',
        Type::NUQ => '2',
        Type::I8 => 'I',
        _ => '?',
    }
}
