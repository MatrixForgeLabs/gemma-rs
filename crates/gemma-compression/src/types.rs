//! Compression type definitions and helpers.

use half::bf16 as BF16;

pub const GEMMA_ENABLE_NUQ: bool = false;

#[repr(C, packed)]
#[derive(Copy, Clone, Default)]
pub struct SfpStream {
    pub byte: u8,
}

impl SfpStream {
    pub const K_MAX: f32 = 1.875;
}

#[repr(C, packed)]
#[derive(Copy, Clone, Default)]
pub struct I8Stream {
    pub i: i8,
}

impl I8Stream {
    pub const K_GROUP_SIZE: usize = 128;

    pub const fn packed_end(capacity: usize) -> usize {
        let num_groups = div_ceil_const(capacity, Self::K_GROUP_SIZE);
        (core::mem::size_of::<BF16>() * num_groups)
            + (core::mem::size_of::<BF16>() * num_groups)
            + capacity
    }
}

#[repr(C, packed)]
#[derive(Copy, Clone, Default)]
pub struct NuqStream {
    pub byte: u8,
}

impl NuqStream {
    pub const K_CLUSTERS: usize = 16;
    pub const K_GROUP_SIZE: usize = 256;

    pub const fn packed_start(capacity: usize) -> usize {
        let num_groups = div_ceil_const(capacity, Self::K_GROUP_SIZE) * Self::K_CLUSTERS;
        round_up_to_const(num_groups, 64)
    }

    pub const fn packed_end(capacity: usize) -> usize {
        let num_groups = div_ceil_const(capacity, Self::K_GROUP_SIZE);
        (Self::K_CLUSTERS * num_groups) + div_ceil_const(capacity, 2)
    }
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Type {
    Unknown = 0,
    F32 = 1,
    BF16 = 2,
    SFP = 3,
    NUQ = 4,
    F64 = 5,
    U32 = 6,
    U64 = 7,
    I8 = 8,
}

impl Default for Type {
    fn default() -> Self {
        Type::Unknown
    }
}

pub const TYPE_STRINGS: [&str; 9] = [
    "unknown", "f32", "bf16", "sfp", "nuq", "f64", "u32", "u64", "i8",
];

pub const TYPE_BITS: [usize; 9] = [
    0,
    8 * core::mem::size_of::<f32>(),
    8 * core::mem::size_of::<BF16>(),
    8 * core::mem::size_of::<SfpStream>(),
    4,
    8 * core::mem::size_of::<f64>(),
    8 * core::mem::size_of::<u32>(),
    8 * core::mem::size_of::<u64>(),
    8 * core::mem::size_of::<I8Stream>(),
];

pub fn enum_valid(value: Type) -> bool {
    (value as usize) < TYPE_STRINGS.len()
}

pub fn type_bits(ty: Type) -> usize {
    TYPE_BITS[ty as usize]
}

pub fn type_name(ty: Type) -> &'static str {
    TYPE_STRINGS[ty as usize]
}

pub fn is_compressed(ty: Type) -> bool {
    matches!(ty, Type::SFP | Type::NUQ | Type::I8)
}

pub fn compressed_array_elements(ty: Type, capacity: usize) -> usize {
    match ty {
        Type::NUQ => NuqStream::packed_end(capacity),
        Type::I8 => I8Stream::packed_end(capacity),
        _ => capacity,
    }
}

#[derive(Copy, Clone)]
pub struct PackedSpan<T> {
    pub ptr: *mut T,
    pub num: usize,
}

impl<T> PackedSpan<T> {
    pub fn bounds_check(&self, packed_ofs: usize, num_accessible: usize) {
        if cfg!(debug_assertions) {
            let required = packed_ofs + num_accessible;
            if self.num < required {
                panic!(
                    "PackedSpan: ofs {}, want {}, req {} > {} packed",
                    packed_ofs, num_accessible, required, self.num
                );
            }
        }
    }
}

pub fn make_span<T>(ptr: *mut T, size: usize) -> PackedSpan<T> {
    PackedSpan { ptr, num: size }
}

pub fn make_const_span<T>(ptr: *mut T, size: usize) -> PackedSpan<T> {
    PackedSpan { ptr, num: size }
}

const fn div_ceil_const(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

const fn round_up_to_const(value: usize, multiple: usize) -> usize {
    if multiple == 0 {
        value
    } else {
        ((value + multiple - 1) / multiple) * multiple
    }
}
