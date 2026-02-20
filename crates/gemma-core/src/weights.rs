//! Placeholder weights container with dynamic tensor map.

use std::collections::HashMap;

use gemma_compression::types::Type;
use gemma_util::allocator::Allocator;
use gemma_util::mat::{MatOwner, MatPadding, MatPtr};

use crate::model_store::ModelStore;
use crate::tensor_info::{extents_from_info, TensorInfoRegistry};

pub struct OwnedTensor {
    pub mat: MatPtr,
    pub owner: MatOwner,
}

pub struct Weights {
    pub store: ModelStore,
    pub tensors: HashMap<String, OwnedTensor>,
}

impl Weights {
    pub fn new(store: ModelStore, registry: &TensorInfoRegistry) -> Self {
        let mut tensors = HashMap::new();
        for name in registry.names() {
            let info = registry.find(&name);
            let extents = extents_from_info(info);
            let mat = MatPtr::new(&name, Type::Unknown, extents);
            tensors.insert(
                name,
                OwnedTensor {
                    mat,
                    owner: MatOwner::new(),
                },
            );
        }
        Self { store, tensors }
    }

    pub fn load_all(&mut self, allocator: &Allocator, padding: MatPadding, default_type: Type) {
        for (name, tensor) in self.tensors.iter_mut() {
            let mat = &mut tensor.mat;
            let range = self.store.find_and_update_mat_ptr(mat, default_type);
            if mat.ty() == Type::Unknown {
                mat.set_type(default_type);
            }
            tensor.owner.allocate_for(mat, allocator, padding);
            let bytes = mat.packed_bytes();
            let ok = if let Some(range) = range {
                self.store.read_range_into(&range, unsafe {
                    std::slice::from_raw_parts_mut(mat.row_bytes(0), bytes)
                })
            } else {
                self.store.read_blob_into(name, unsafe {
                    std::slice::from_raw_parts_mut(mat.row_bytes(0), bytes)
                })
            };
            if !ok {
                // Leave tensor allocated but zeroed.
                continue;
            }
        }
    }
}
