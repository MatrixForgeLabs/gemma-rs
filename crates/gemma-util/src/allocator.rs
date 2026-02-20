//! Simplified allocator with alignment support.

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

use crate::topology::BoundedTopology;

#[derive(Debug)]
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    layout: Layout,
    len: usize,
}

impl AlignedBuffer {
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

#[derive(Clone, Debug)]
pub struct CacheInfo {
    line_bytes: usize,
    vector_bytes: usize,
    step_bytes: usize,
    l1_bytes: usize,
    l2_bytes: usize,
    l3_bytes: usize,
}

impl CacheInfo {
    pub fn new(_topology: &BoundedTopology) -> Self {
        let line_bytes = 64;
        let vector_bytes = 32;
        let step_bytes = line_bytes.max(vector_bytes);
        Self {
            line_bytes,
            vector_bytes,
            step_bytes,
            l1_bytes: 32 << 10,
            l2_bytes: 256 << 10,
            l3_bytes: 1024 << 10,
        }
    }

    pub fn line_bytes(&self) -> usize {
        self.line_bytes
    }

    pub fn max_line_bytes() -> usize {
        256
    }

    pub fn vector_bytes(&self) -> usize {
        self.vector_bytes
    }

    pub fn step_bytes(&self) -> usize {
        self.step_bytes
    }

    pub fn l1_bytes(&self) -> usize {
        self.l1_bytes
    }

    pub fn l2_bytes(&self) -> usize {
        self.l2_bytes
    }

    pub fn l3_bytes(&self) -> usize {
        self.l3_bytes
    }
}

#[derive(Clone, Debug)]
pub struct Allocator {
    line_bytes: usize,
    base_page_bytes: usize,
    total_mib: usize,
    quantum_bytes: usize,
    should_bind: bool,
}

impl Allocator {
    pub fn new(_topology: &BoundedTopology, cache_info: &CacheInfo, _enable_bind: bool) -> Self {
        let base_page_bytes = detect_page_size().unwrap_or(4096);
        let total_mib = detect_total_mib(base_page_bytes).unwrap_or(0);
        let quantum_bytes = cache_info.step_bytes();
        Self {
            line_bytes: cache_info.line_bytes(),
            base_page_bytes,
            total_mib,
            quantum_bytes,
            should_bind: false,
        }
    }

    pub fn line_bytes(&self) -> usize {
        self.line_bytes
    }

    pub fn base_page_bytes(&self) -> usize {
        self.base_page_bytes
    }

    pub fn quantum_bytes(&self) -> usize {
        self.quantum_bytes
    }

    pub fn total_mib(&self) -> usize {
        self.total_mib
    }

    pub fn free_mib(&self) -> usize {
        detect_free_mib(self.base_page_bytes).unwrap_or(self.total_mib)
    }

    pub fn should_bind(&self) -> bool {
        self.should_bind
    }

    pub fn alloc_bytes(&self, bytes: usize) -> AlignedBuffer {
        let alignment = self.quantum_bytes.max(16);
        let layout = Layout::from_size_align(bytes, alignment).expect("bad layout");
        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr).expect("alloc failed");
        AlignedBuffer {
            ptr,
            layout,
            len: bytes,
        }
    }
}

fn detect_page_size() -> Option<usize> {
    #[cfg(unix)]
    unsafe {
        let ret = libc::sysconf(libc::_SC_PAGESIZE);
        if ret > 0 {
            Some(ret as usize)
        } else {
            None
        }
    }
    #[cfg(not(unix))]
    {
        None
    }
}

fn detect_total_mib(page_bytes: usize) -> Option<usize> {
    #[cfg(unix)]
    unsafe {
        let pages = libc::sysconf(libc::_SC_PHYS_PAGES);
        if pages > 0 {
            Some((pages as usize * page_bytes) >> 20)
        } else {
            None
        }
    }
    #[cfg(not(unix))]
    {
        Some(0)
    }
}

fn detect_free_mib(page_bytes: usize) -> Option<usize> {
    #[cfg(unix)]
    unsafe {
        let pages = libc::sysconf(libc::_SC_AVPHYS_PAGES);
        if pages > 0 {
            Some((pages as usize * page_bytes) >> 20)
        } else {
            None
        }
    }
    #[cfg(not(unix))]
    {
        None
    }
}
