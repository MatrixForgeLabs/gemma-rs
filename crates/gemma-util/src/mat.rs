//! Tensor metadata and in-memory representation (partial port).

use core::marker::PhantomData;
use core::ptr::{copy_nonoverlapping, write_bytes};

use gemma_compression::types::{
    compressed_array_elements, type_bits, type_name, Type, GEMMA_ENABLE_NUQ,
};
use gemma_io::fields::{Fields, FieldsVisitor};

use crate::allocator::{AlignedBuffer, Allocator};
use crate::basics::{Extents2D, BF16};

pub struct RowPtrs<T> {
    row_ptrs: *mut *mut u8,
    r0: u32,
    c0: u32,
    _marker: PhantomData<T>,
}

impl<T> RowPtrs<T> {
    pub fn new(row_ptrs: *mut *mut u8) -> Self {
        Self {
            row_ptrs,
            r0: 0,
            c0: 0,
            _marker: PhantomData,
        }
    }

    pub fn view(&self, r: usize, c: usize, _cols: usize) -> Self {
        Self {
            row_ptrs: self.row_ptrs,
            r0: self.r0 + r as u32,
            c0: self.c0 + c as u32,
            _marker: PhantomData,
        }
    }

    pub fn row(&self, row_idx: usize) -> *mut T {
        unsafe {
            let base = *self.row_ptrs.add(self.r0 as usize + row_idx);
            (base as *mut T).add(self.c0 as usize)
        }
    }
}

pub type RowPtrsBF = RowPtrs<BF16>;

pub struct MatPtr {
    name: String,
    ty: Type,
    element_bytes: u32,
    num_elements: u32,
    private_rows: u32,
    cols: u32,
    override_rows: u32,
    ptr: *mut u8,
    row_ptrs: *mut *mut u8,
    stride: u32,
    scale: f32,
}

impl MatPtr {
    pub fn new(name: &str, ty: Type, extents: Extents2D) -> Self {
        let mut mat = Self {
            name: String::new(),
            ty,
            element_bytes: 0,
            num_elements: 0,
            private_rows: extents.rows as u32,
            cols: extents.cols as u32,
            override_rows: 0,
            ptr: core::ptr::null_mut(),
            row_ptrs: core::ptr::null_mut(),
            stride: 0,
            scale: 1.0,
        };
        mat.set_name(name);
        mat.set_type(ty);
        mat.set_ptr(core::ptr::null_mut(), extents.cols);
        mat
    }

    pub fn set_ptr(&mut self, ptr: *mut u8, stride: usize) {
        assert!(stride >= self.cols as usize);
        self.ptr = ptr;
        self.stride = stride as u32;
        assert!(self.row_ptrs.is_null(), "Do not call after attach_row_ptrs");

        if self.ty == Type::NUQ {
            assert!(GEMMA_ENABLE_NUQ, "Set GEMMA_ENABLE_NUQ=1");
            assert!(self.is_packed());
        }
    }

    pub fn has_ptr(&self) -> bool {
        !self.ptr.is_null()
    }

    pub fn attach_row_ptrs(&mut self, row_ptrs: *mut *mut u8) {
        self.row_ptrs = row_ptrs;
    }

    pub fn allocate_and_attach_row_ptrs(&mut self, storage: &mut Vec<Vec<*mut u8>>) {
        if !self.has_ptr() {
            return;
        }
        let mut row_ptrs = vec![core::ptr::null_mut(); self.rows()];
        for r in 0..self.rows() {
            row_ptrs[r] = self.row_bytes(r);
        }
        self.attach_row_ptrs(row_ptrs.as_mut_ptr());
        storage.push(row_ptrs);
    }

    pub fn get_row_ptrs(&self) -> *mut *mut u8 {
        self.row_ptrs
    }

    pub fn is_packed(&self) -> bool {
        self.stride == self.cols || self.rows() == 1
    }

    pub fn packed_bytes(&self) -> usize {
        assert!(self.is_packed(), "{}", self.name);
        self.num_elements as usize * self.element_bytes as usize
    }

    pub fn row_bytes(&self, row: usize) -> *mut u8 {
        assert!(row < self.rows());
        unsafe {
            self.ptr
                .add(row * (self.stride as usize * self.element_bytes as usize))
        }
    }

    pub fn ty(&self) -> Type {
        self.ty
    }

    pub fn set_type(&mut self, ty: Type) {
        self.ty = ty;
        if ty == Type::Unknown {
            self.element_bytes = 0;
            self.num_elements = 0;
            return;
        }
        self.element_bytes = div_ceil(type_bits(ty), 8) as u32;
        self.num_elements = Self::compute_num_elements(ty, self.extents()) as u32;
        assert!(self.element_bytes != 0 && self.element_bytes <= 16);
    }

    pub fn rows(&self) -> usize {
        if self.override_rows == 0 {
            self.private_rows as usize
        } else {
            self.override_rows as usize
        }
    }

    pub fn cols(&self) -> usize {
        self.cols as usize
    }

    pub fn extents(&self) -> Extents2D {
        Extents2D::new(self.rows(), self.cols())
    }

    pub fn is_empty(&self) -> bool {
        self.rows() == 0 || self.cols() == 0
    }

    pub fn same_shape(&self, other: &MatPtr) -> bool {
        self.rows() == other.rows() && self.cols == other.cols
    }

    pub fn override_rows(&mut self, rows: usize) {
        if rows > self.private_rows as usize {
            panic!(
                "{}: rows {} > private_rows {}",
                self.name, rows, self.private_rows
            );
        }
        self.override_rows = rows as u32;
    }

    pub fn stride(&self) -> usize {
        self.stride as usize
    }

    pub fn element_bytes(&self) -> usize {
        self.element_bytes as usize
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale;
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn set_name(&mut self, name: &str) {
        assert!(name.len() <= 16, "{}", name);
        self.name = name.to_string();
    }

    fn compute_num_elements(ty: Type, extents: Extents2D) -> usize {
        let num_elements = extents.area();
        compressed_array_elements(ty, num_elements)
    }
}

impl Clone for MatPtr {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            ty: self.ty,
            element_bytes: self.element_bytes,
            num_elements: self.num_elements,
            private_rows: self.private_rows,
            cols: self.cols,
            override_rows: self.override_rows,
            ptr: self.ptr,
            row_ptrs: self.row_ptrs,
            stride: self.stride,
            scale: self.scale,
        }
    }
}

impl Fields for MatPtr {
    fn name(&self) -> &str {
        self.name()
    }

    fn visit_fields(&mut self, visitor: &mut dyn FieldsVisitor) {
        let mut name = self.name.clone();
        visitor.visit_string(&mut name);
        self.name = name;

        let mut ty_u32 = self.ty as u32;
        visitor.visit_u32(&mut ty_u32);
        self.ty = match ty_u32 {
            0 => Type::Unknown,
            1 => Type::F32,
            2 => Type::BF16,
            3 => Type::SFP,
            4 => Type::NUQ,
            5 => Type::F64,
            6 => Type::U32,
            7 => Type::U64,
            8 => Type::I8,
            _ => {
                visitor.notify_invalid(&format!("Invalid enum {}", ty_u32));
                Type::Unknown
            }
        };
        self.set_type(self.ty);

        let mut element_bytes = self.element_bytes;
        visitor.visit_u32(&mut element_bytes);
        self.element_bytes = element_bytes;

        let mut num_elements = self.num_elements;
        visitor.visit_u32(&mut num_elements);
        self.num_elements = num_elements;

        let mut private_rows = self.private_rows;
        visitor.visit_u32(&mut private_rows);
        self.private_rows = private_rows;

        let mut cols = self.cols;
        visitor.visit_u32(&mut cols);
        self.cols = cols;

        let mut scale = self.scale;
        visitor.visit_f32(&mut scale);
        self.scale = scale;

        let mut stride = self.stride;
        visitor.visit_u32(&mut stride);
        self.stride = stride;
    }
}

pub struct MatPtrT<T> {
    pub base: MatPtr,
    _marker: PhantomData<T>,
}

impl<T> MatPtrT<T> {
    pub fn new(name: &str, extents: Extents2D, ty: Type) -> Self {
        Self {
            base: MatPtr::new(name, ty, extents),
            _marker: PhantomData,
        }
    }

    pub fn row(&self, row: usize) -> *mut T {
        self.base.row_bytes(row) as *mut T
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MatPadding {
    Packed,
    Odd,
}

pub fn stride(padding: MatPadding, cols: usize, element_bytes: usize, line_bytes: usize) -> usize {
    match padding {
        MatPadding::Packed => cols,
        MatPadding::Odd => {
            let line_elems = (line_bytes / element_bytes).max(1);
            let mut stride = round_up(cols, line_elems);
            let lines = stride / line_elems;
            if lines % 2 == 0 {
                stride += line_elems;
            }
            stride
        }
    }
}

pub struct MatOwner {
    storage: Option<AlignedBuffer>,
}

impl MatOwner {
    pub fn new() -> Self {
        Self { storage: None }
    }

    pub fn allocate_for(&mut self, mat: &mut MatPtr, allocator: &Allocator, padding: MatPadding) {
        if mat.ty() == Type::NUQ || mat.ty() == Type::I8 {
            let bytes = mat.num_elements as usize * mat.element_bytes();
            let storage = allocator.alloc_bytes(bytes);
            mat.set_ptr(storage.as_ptr(), mat.cols());
            self.storage = Some(storage);
            return;
        }

        let stride = stride(
            padding,
            mat.cols(),
            mat.element_bytes(),
            allocator.line_bytes(),
        );
        let elements = mat.rows() * stride;
        let bytes = elements * mat.element_bytes();
        let storage = allocator.alloc_bytes(bytes);
        mat.set_ptr(storage.as_ptr(), stride);
        self.storage = Some(storage);
    }

    pub fn storage_len(&self) -> usize {
        self.storage.as_ref().map(|b| b.len()).unwrap_or(0)
    }
}

pub struct MatStorageT<T> {
    pub mat: MatPtrT<T>,
    _owner: MatOwner,
}

impl<T> MatStorageT<T> {
    pub fn new(
        name: &str,
        extents: Extents2D,
        allocator: &Allocator,
        padding: MatPadding,
        ty: Type,
    ) -> Self {
        let mut mat = MatPtrT::new(name, extents, ty);
        let mut owner = MatOwner::new();
        if extents.area() != 0 {
            owner.allocate_for(&mut mat.base, allocator, padding);
        }
        Self { mat, _owner: owner }
    }

    pub fn new_1d(name: &str, cols: usize, allocator: &Allocator, ty: Type) -> Self {
        Self::new(
            name,
            Extents2D::new(1, cols),
            allocator,
            MatPadding::Packed,
            ty,
        )
    }
}

pub struct MatFactory {
    name: String,
    extents: Extents2D,
    allocator: *const Allocator,
    padding: MatPadding,
}

impl MatFactory {
    pub fn new(
        name: &str,
        rows: usize,
        cols: usize,
        allocator: &Allocator,
        padding: MatPadding,
    ) -> Self {
        Self {
            name: name.to_string(),
            extents: Extents2D::new(rows, cols),
            allocator,
            padding,
        }
    }

    pub fn make<T>(&self, ty: Type) -> MatStorageT<T> {
        let allocator = unsafe { &*self.allocator };
        MatStorageT::new(&self.name, self.extents, allocator, self.padding, ty)
    }
}

#[repr(C, packed)]
pub struct StridedView<T> {
    row0: *mut T,
    cols: u32,
    stride: u32,
}

impl<T> StridedView<T> {
    pub fn new(row0: *mut T, cols: usize, stride: usize) -> Self {
        if cfg!(debug_assertions) && stride < cols {
            panic!("stride {} < cols {}", stride, cols);
        }
        Self {
            row0,
            cols: cols as u32,
            stride: stride as u32,
        }
    }

    pub fn from_mat(mat: &MatPtrT<T>, r: usize, c: usize, cols: usize) -> Self {
        let row = unsafe { mat.row(r).add(c) };
        Self::new(row, cols, mat.base.stride())
    }

    pub fn view(&self, r: usize, c: usize, cols: usize) -> Self {
        let row = unsafe { self.row(r).add(c) };
        Self::new(row, cols, self.stride())
    }

    pub fn row(&self, r: usize) -> *mut T {
        unsafe { self.row0.add(self.stride as usize * r) }
    }

    pub fn cols(&self) -> usize {
        self.cols as usize
    }

    pub fn stride(&self) -> usize {
        self.stride as usize
    }

    pub fn set_stride(&mut self, stride: usize) {
        if cfg!(debug_assertions) && stride < self.cols() {
            panic!("stride {} < cols {}", stride, self.cols());
        }
        self.stride = stride as u32;
    }
}

pub fn copy_mat(from: &MatPtr, to: &mut MatPtr) {
    assert!(from.same_shape(to));
    assert!(from.ty() == to.ty());
    let bytes_per_row = from.cols() * from.element_bytes();
    if from.is_packed() && to.is_packed() {
        unsafe {
            copy_nonoverlapping(from.row_bytes(0), to.row_bytes(0), from.packed_bytes());
        }
        return;
    }
    for r in 0..from.rows() {
        unsafe {
            copy_nonoverlapping(from.row_bytes(r), to.row_bytes(r), bytes_per_row);
        }
    }
}

pub fn zero_init(mat: &mut MatPtr) {
    let bytes_per_row = mat.cols() * mat.element_bytes();
    if mat.is_packed() {
        unsafe {
            write_bytes(mat.row_bytes(0), 0, mat.packed_bytes());
        }
        return;
    }
    for r in 0..mat.rows() {
        unsafe {
            write_bytes(mat.row_bytes(r), 0, bytes_per_row);
        }
    }
}

fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

fn round_up(value: usize, multiple: usize) -> usize {
    if multiple == 0 {
        value
    } else {
        ((value + multiple - 1) / multiple) * multiple
    }
}

#[allow(dead_code)]
pub fn type_name_for(ty: Type) -> &'static str {
    type_name(ty)
}
