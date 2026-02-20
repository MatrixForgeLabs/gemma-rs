//! Core basic types and helpers.

use core::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

pub const K_MAX_BATCH_SIZE: usize = 4096;

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Tristate {
    False = 0,
    True = 1,
    Default = -1,
}

impl Default for Tristate {
    fn default() -> Self {
        Tristate::Default
    }
}

impl Tristate {
    pub fn as_str(self) -> &'static str {
        match self {
            Tristate::False => "false",
            Tristate::True => "true",
            Tristate::Default => "default",
        }
    }
}

pub type BF16 = half::bf16;

#[repr(C, packed)]
#[derive(Copy, Clone, Default)]
pub struct TokenAndProb {
    pub token: i32,
    pub prob: f32,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct Extents2D {
    pub rows: usize,
    pub cols: usize,
}

impl Extents2D {
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }

    pub const fn area(self) -> usize {
        self.rows * self.cols
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct IndexRange {
    pub begin: usize,
    pub end: usize,
}

impl IndexRange {
    pub fn new(begin: usize, end: usize) -> Self {
        debug_assert!(begin < end);
        Self { begin, end }
    }

    pub fn num(self) -> usize {
        self.end - self.begin
    }

    pub fn contains(self, other: IndexRange) -> bool {
        other.begin >= self.begin && other.end <= self.end
    }

    pub fn iter(self) -> IndexRangeIter {
        IndexRangeIter {
            cur: self.begin,
            end: self.end,
        }
    }
}

pub struct IndexRangeIter {
    cur: usize,
    end: usize,
}

impl Iterator for IndexRangeIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur >= self.end {
            None
        } else {
            let out = self.cur;
            self.cur += 1;
            Some(out)
        }
    }
}

pub fn make_index_range(begin: usize, end: usize, max_size: usize) -> IndexRange {
    IndexRange::new(begin, core::cmp::min(begin + max_size, end))
}

pub type Logits<'a> = &'a mut [f32];

pub fn maybe_check_initialized(_ptr: *const u8, _size: usize) {}

pub fn maybe_print_initialized(_ptr: *const u8, _size: usize) {}

pub fn maybe_test_initialized(_ptr: *const u8, _size: usize) -> isize {
    0
}

#[derive(Clone)]
pub struct AesCtrEngine {
    key: [u64; 2 * (1 + 5)],
}

impl AesCtrEngine {
    pub fn new(deterministic: bool) -> Self {
        let mut key = [0u64; 12];
        // Pi-based nothing up my sleeve numbers from Randen.
        key[0] = 0x243F6A88_85A308D3u64;
        key[1] = 0x13198A2E_03707344u64;

        if !deterministic {
            let mut buf = [0u8; 16];
            if getrandom::getrandom(&mut buf).is_ok() {
                key[0] = u64::from_le_bytes(buf[0..8].try_into().unwrap());
                key[1] = u64::from_le_bytes(buf[8..16].try_into().unwrap());
            } else {
                let addr = (&key as *const _ as usize) as u64;
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;
                key[0] ^= addr;
                key[1] ^= now;
            }
        }

        for i in 0..5 {
            key[2 + 2 * i] = key[2 * i + 1].wrapping_add(0xA4093822_299F31D0u64);
            key[2 + 2 * i + 1] = key[2 * i].wrapping_add(0x082EFA98_EC4E6C89u64);
        }

        Self { key }
    }

    pub fn generate(&self, stream: u64, counter: u64) -> u64 {
        let mut state = [0u8; 16];
        state[0..8].copy_from_slice(&counter.to_le_bytes());
        state[8..16].copy_from_slice(&stream.to_le_bytes());

        let round0 = Self::round_key_bytes(self.key[0], self.key[1]);
        for i in 0..16 {
            state[i] ^= round0[i];
        }

        for r in 0..5 {
            let rk = Self::round_key_bytes(self.key[2 + 2 * r], self.key[3 + 2 * r]);
            state = aes_round(state, rk);
        }

        u64::from_le_bytes(state[0..8].try_into().unwrap())
    }

    fn round_key_bytes(k0: u64, k1: u64) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[0..8].copy_from_slice(&k0.to_le_bytes());
        out[8..16].copy_from_slice(&k1.to_le_bytes());
        out
    }
}

#[derive(Clone, Default)]
pub struct RngStream<'a> {
    engine: Option<&'a AesCtrEngine>,
    stream: u64,
    counter: u64,
}

impl<'a> RngStream<'a> {
    pub fn new(engine: &'a AesCtrEngine, stream: u64) -> Self {
        Self {
            engine: Some(engine),
            stream,
            counter: 0,
        }
    }

    pub fn reset(&mut self, engine: &'a AesCtrEngine, stream: u64) {
        self.engine = Some(engine);
        self.stream = stream;
        self.counter = 0;
    }

    pub fn next_u64(&mut self) -> u64 {
        match self.engine {
            Some(engine) => {
                let v = engine.generate(self.stream, self.counter);
                self.counter = self.counter.wrapping_add(1);
                v
            }
            None => 0,
        }
    }
}

impl<'a> Iterator for RngStream<'a> {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        Some(self.next_u64())
    }
}

impl<'a> fmt::Debug for RngStream<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RngStream")
            .field("stream", &self.stream)
            .field("counter", &self.counter)
            .finish()
    }
}

const SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

fn aes_round(mut state: [u8; 16], round_key: [u8; 16]) -> [u8; 16] {
    for b in &mut state {
        *b = SBOX[*b as usize];
    }

    // ShiftRows
    let mut tmp = state;
    tmp[1] = state[5];
    tmp[5] = state[9];
    tmp[9] = state[13];
    tmp[13] = state[1];

    tmp[2] = state[10];
    tmp[6] = state[14];
    tmp[10] = state[2];
    tmp[14] = state[6];

    tmp[3] = state[15];
    tmp[7] = state[3];
    tmp[11] = state[7];
    tmp[15] = state[11];

    // MixColumns
    for c in 0..4 {
        let i = c * 4;
        let a0 = tmp[i];
        let a1 = tmp[i + 1];
        let a2 = tmp[i + 2];
        let a3 = tmp[i + 3];
        let t = a0 ^ a1 ^ a2 ^ a3;
        let x = a0;
        tmp[i] = a0 ^ t ^ xtime(a0 ^ a1);
        tmp[i + 1] = a1 ^ t ^ xtime(a1 ^ a2);
        tmp[i + 2] = a2 ^ t ^ xtime(a2 ^ a3);
        tmp[i + 3] = a3 ^ t ^ xtime(a3 ^ x);
    }

    for i in 0..16 {
        tmp[i] ^= round_key[i];
    }

    tmp
}

fn xtime(x: u8) -> u8 {
    (x << 1) ^ if x & 0x80 != 0 { 0x1b } else { 0 }
}
