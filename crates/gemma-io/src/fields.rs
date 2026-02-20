//! Field serialization compatible with the C++ implementation.

use std::fmt;

pub type SerializedSpan<'a> = &'a [u32];

pub trait Fields {
    fn name(&self) -> &str;
    fn visit_fields(&mut self, visitor: &mut dyn FieldsVisitor);
}

pub trait FieldsVisitor {
    fn any_invalid(&self) -> bool;
    fn notify_invalid(&mut self, msg: &str);

    fn visit_u32(&mut self, value: &mut u32);
    fn visit_i32(&mut self, value: &mut i32);
    fn visit_u64(&mut self, value: &mut u64);
    fn visit_f32(&mut self, value: &mut f32);
    fn visit_string(&mut self, value: &mut String);
    fn visit_fields(&mut self, fields: &mut dyn Fields);

    fn skip_field(&mut self) -> bool {
        self.any_invalid()
    }

    fn is_reading(&self) -> bool {
        false
    }

    fn visit_bool(&mut self, value: &mut bool) {
        if self.skip_field() {
            return;
        }
        let mut u32_val = if *value { 1u32 } else { 0u32 };
        self.visit_u32(&mut u32_val);
        if u32_val > 1 {
            self.notify_invalid(&format!("Invalid bool {}", u32_val));
            return;
        }
        *value = u32_val == 1;
    }

    fn visit_vec_u32(&mut self, value: &mut Vec<u32>) {
        if self.skip_field() {
            return;
        }

        let mut num = value.len() as u32;
        self.visit_u32(&mut num);
        if num > 64 * 1024 {
            self.notify_invalid(&format!("Vector too long {}", num));
            return;
        }

        if self.is_reading() {
            value.resize(num as usize, 0);
        }

        for item in value.iter_mut() {
            self.visit_u32(item);
        }
    }

    fn visit_vec_i32(&mut self, value: &mut Vec<i32>) {
        if self.skip_field() {
            return;
        }

        let mut num = value.len() as u32;
        self.visit_u32(&mut num);
        if num > 64 * 1024 {
            self.notify_invalid(&format!("Vector too long {}", num));
            return;
        }

        if self.is_reading() {
            value.resize(num as usize, 0);
        }

        for item in value.iter_mut() {
            self.visit_i32(item);
        }
    }

    fn visit_vec_u64(&mut self, value: &mut Vec<u64>) {
        if self.skip_field() {
            return;
        }

        let mut num = value.len() as u32;
        self.visit_u32(&mut num);
        if num > 64 * 1024 {
            self.notify_invalid(&format!("Vector too long {}", num));
            return;
        }

        if self.is_reading() {
            value.resize(num as usize, 0);
        }

        for item in value.iter_mut() {
            self.visit_u64(item);
        }
    }

    fn visit_vec_f32(&mut self, value: &mut Vec<f32>) {
        if self.skip_field() {
            return;
        }

        let mut num = value.len() as u32;
        self.visit_u32(&mut num);
        if num > 64 * 1024 {
            self.notify_invalid(&format!("Vector too long {}", num));
            return;
        }

        if self.is_reading() {
            value.resize(num as usize, 0.0);
        }

        for item in value.iter_mut() {
            self.visit_f32(item);
        }
    }

    fn visit_vec_string(&mut self, value: &mut Vec<String>) {
        if self.skip_field() {
            return;
        }

        let mut num = value.len() as u32;
        self.visit_u32(&mut num);
        if num > 64 * 1024 {
            self.notify_invalid(&format!("Vector too long {}", num));
            return;
        }

        if self.is_reading() {
            value.resize_with(num as usize, String::new);
        }

        for item in value.iter_mut() {
            self.visit_string(item);
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ReadResult {
    pub pos: usize,
    pub missing_fields: u32,
    pub extra_u32: u32,
}

impl ReadResult {
    pub fn new(pos: usize) -> Self {
        Self {
            pos,
            missing_fields: 0,
            extra_u32: 0,
        }
    }
}

pub fn print_fields(fields: &mut dyn Fields) {
    let mut visitor = PrintVisitor::default();
    visitor.visit_fields(fields);
}

pub fn read_fields(fields: &mut dyn Fields, span: SerializedSpan, pos: usize) -> ReadResult {
    let mut visitor = ReadVisitor::new(span, pos);
    visitor.visit_fields(fields);
    visitor.result()
}

pub fn append_to(fields: &mut dyn Fields, storage: &mut Vec<u32>) -> bool {
    let mut visitor = WriteVisitor::new(storage);
    visitor.visit_fields(fields);
    !visitor.any_invalid
}

pub fn write_fields(fields: &mut dyn Fields) -> Vec<u32> {
    let mut storage = Vec::new();
    if !append_to(fields, &mut storage) {
        storage.clear();
    }
    storage
}

#[derive(Default)]
struct PrintVisitor {
    indent: String,
    any_invalid: bool,
}

impl PrintVisitor {
    fn indent_line(&self) -> &str {
        &self.indent
    }
}

impl FieldsVisitor for PrintVisitor {
    fn any_invalid(&self) -> bool {
        self.any_invalid
    }

    fn notify_invalid(&mut self, msg: &str) {
        eprintln!("{}", msg);
        self.any_invalid = true;
    }

    fn visit_u32(&mut self, value: &mut u32) {
        eprintln!("{}U32 {}", self.indent_line(), value);
    }

    fn visit_i32(&mut self, value: &mut i32) {
        eprintln!("{}I32 {}", self.indent_line(), value);
    }

    fn visit_u64(&mut self, value: &mut u64) {
        eprintln!("{}U64 {}", self.indent_line(), value);
    }

    fn visit_f32(&mut self, value: &mut f32) {
        eprintln!("{}F32 {}", self.indent_line(), value);
    }

    fn visit_string(&mut self, value: &mut String) {
        eprintln!("{}Str {}", self.indent_line(), value);
    }

    fn visit_fields(&mut self, fields: &mut dyn Fields) {
        eprintln!("{}{}", self.indent_line(), fields.name());
        self.indent.push_str("  ");
        fields.visit_fields(self);
        self.indent.truncate(self.indent.len().saturating_sub(2));
    }
}

struct ReadVisitor<'a> {
    span: SerializedSpan<'a>,
    result: ReadResult,
    end: Vec<usize>,
    any_invalid: bool,
}

impl<'a> ReadVisitor<'a> {
    fn new(span: SerializedSpan<'a>, pos: usize) -> Self {
        Self {
            span,
            result: ReadResult::new(pos),
            end: Vec::new(),
            any_invalid: false,
        }
    }

    fn check_f32(&mut self, value: f32) {
        if value.is_nan() || value.is_infinite() {
            self.notify_invalid(&format!("Invalid float {}", value));
        }
    }

    fn check_string_length(&mut self, num_u32: u32) -> bool {
        if num_u32 > 64 {
            self.notify_invalid(&format!("String num_u32={} too large", num_u32));
            return false;
        }
        true
    }

    fn check_string_u32(&mut self, u32_val: u32, i: u32, num_u32: u32) -> bool {
        if u32_val == 0 || (u32_val & 0x8080_8080) != 0 {
            self.notify_invalid(&format!(
                "Invalid characters {:x} at {} of {}",
                u32_val, i, num_u32
            ));
            return false;
        }
        true
    }

    fn result(&mut self) -> ReadResult {
        if self.any_invalid {
            self.result.pos = 0;
        }
        self.result
    }
}

impl<'a> FieldsVisitor for ReadVisitor<'a> {
    fn any_invalid(&self) -> bool {
        self.any_invalid
    }

    fn notify_invalid(&mut self, msg: &str) {
        eprintln!("{}", msg);
        self.any_invalid = true;
    }

    fn visit_u32(&mut self, value: &mut u32) {
        if self.skip_field() {
            return;
        }
        *value = self.span[self.result.pos];
        self.result.pos += 1;
    }

    fn visit_i32(&mut self, value: &mut i32) {
        if self.skip_field() {
            return;
        }
        *value = self.span[self.result.pos] as i32;
        self.result.pos += 1;
    }

    fn visit_u64(&mut self, value: &mut u64) {
        if self.skip_field() {
            return;
        }
        let mut lower = *value as u32;
        self.visit_u32(&mut lower);
        let mut upper = (*value >> 32) as u32;
        self.visit_u32(&mut upper);
        *value = lower as u64 | ((upper as u64) << 32);
    }

    fn visit_f32(&mut self, value: &mut f32) {
        if self.skip_field() {
            return;
        }
        let mut bits = value.to_bits();
        self.visit_u32(&mut bits);
        *value = f32::from_bits(bits);
        self.check_f32(*value);
    }

    fn visit_string(&mut self, value: &mut String) {
        if self.skip_field() {
            return;
        }

        let mut num_u32 = 0u32;
        self.visit_u32(&mut num_u32);
        if !self.check_string_length(num_u32) {
            return;
        }

        if let Some(end) = self.end.last().copied() {
            if self.result.pos + num_u32 as usize > end {
                self.notify_invalid(&format!(
                    "Invalid string: pos {} + num_u32 {} > end {}",
                    self.result.pos, num_u32, end
                ));
                return;
            }
        }

        let mut bytes = vec![0u8; num_u32 as usize * 4];
        for i in 0..num_u32 {
            let mut u32_val = 0u32;
            self.visit_u32(&mut u32_val);
            self.check_string_u32(u32_val, i, num_u32);
            bytes[(i as usize) * 4..(i as usize + 1) * 4].copy_from_slice(&u32_val.to_le_bytes());
        }

        while bytes.last() == Some(&0) {
            bytes.pop();
        }
        *value = String::from_utf8_lossy(&bytes).to_string();
    }

    fn visit_fields(&mut self, fields: &mut dyn Fields) {
        self.end.push(self.span.len());

        if self.skip_field() {
            self.end.pop();
            return;
        }

        let mut num_u32 = 0u32;
        self.visit_u32(&mut num_u32);
        if self.result.pos + num_u32 as usize > self.span.len() {
            self.notify_invalid(&format!(
                "Invalid IFields: pos {} + num_u32 {} > size {}",
                self.result.pos,
                num_u32,
                self.span.len()
            ));
            return;
        }

        let end_pos = self.result.pos + num_u32 as usize;
        if let Some(last) = self.end.last_mut() {
            *last = end_pos;
        }

        fields.visit_fields(self);

        if let Some(last) = self.end.last().copied() {
            if self.result.pos <= last {
                self.result.extra_u32 += (last - self.result.pos) as u32;
            }
        }
        self.end.pop();
    }

    fn skip_field(&mut self) -> bool {
        if self.any_invalid {
            return true;
        }

        if let Some(end) = self.end.last().copied() {
            if self.result.pos >= end {
                self.result.missing_fields += 1;
                return true;
            }
        }

        false
    }

    fn is_reading(&self) -> bool {
        true
    }
}

struct WriteVisitor<'a> {
    storage: &'a mut Vec<u32>,
    any_invalid: bool,
}

impl<'a> WriteVisitor<'a> {
    fn new(storage: &'a mut Vec<u32>) -> Self {
        Self {
            storage,
            any_invalid: false,
        }
    }

    fn check_f32(&mut self, value: f32) {
        if value.is_nan() || value.is_infinite() {
            self.notify_invalid(&format!("Invalid float {}", value));
        }
    }

    fn check_string_length(&mut self, num_u32: u32) -> bool {
        if num_u32 > 64 {
            self.notify_invalid(&format!("String num_u32={} too large", num_u32));
            return false;
        }
        true
    }

    fn check_string_u32(&mut self, u32_val: u32, i: u32, num_u32: u32) -> bool {
        if u32_val == 0 || (u32_val & 0x8080_8080) != 0 {
            self.notify_invalid(&format!(
                "Invalid characters {:x} at {} of {}",
                u32_val, i, num_u32
            ));
            return false;
        }
        true
    }
}

impl<'a> FieldsVisitor for WriteVisitor<'a> {
    fn any_invalid(&self) -> bool {
        self.any_invalid
    }

    fn notify_invalid(&mut self, msg: &str) {
        eprintln!("{}", msg);
        self.any_invalid = true;
    }

    fn visit_u32(&mut self, value: &mut u32) {
        self.storage.push(*value);
    }

    fn visit_i32(&mut self, value: &mut i32) {
        self.storage.push(*value as u32);
    }

    fn visit_u64(&mut self, value: &mut u64) {
        self.storage.push(*value as u32);
        self.storage.push((*value >> 32) as u32);
    }

    fn visit_f32(&mut self, value: &mut f32) {
        self.storage.push(value.to_bits());
        self.check_f32(*value);
    }

    fn visit_string(&mut self, value: &mut String) {
        let num_u32 = div_ceil(value.len(), 4) as u32;
        if !self.check_string_length(num_u32) {
            return;
        }
        let mut num = num_u32;
        self.visit_u32(&mut num);

        let bytes = value.as_bytes();
        let num_whole = bytes.len() / 4;
        for i in 0..num_whole {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&bytes[i * 4..i * 4 + 4]);
            let u32_val = u32::from_le_bytes(buf);
            if !self.check_string_u32(u32_val, i as u32, num_u32) {
                return;
            }
            self.storage.push(u32_val);
        }

        let remainder = bytes.len() - num_whole * 4;
        if remainder != 0 {
            let mut u32_val = 0u32;
            for i in 0..remainder {
                u32_val |= (bytes[num_whole * 4 + i] as u32) << (i * 8);
            }
            if !self.check_string_u32(u32_val, num_whole as u32, num_u32) {
                return;
            }
            self.storage.push(u32_val);
        }
    }

    fn visit_fields(&mut self, fields: &mut dyn Fields) {
        let pos_before = self.storage.len();
        self.storage.push(0);
        fields.visit_fields(self);
        let num_u32 = self.storage.len() - pos_before;
        self.storage[pos_before] = (num_u32 - 1) as u32;
    }
}

fn div_ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

impl fmt::Debug for dyn Fields {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}
