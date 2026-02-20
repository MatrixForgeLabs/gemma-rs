use gemma_io::fields::{read_fields, write_fields, Fields, FieldsVisitor};

#[derive(Default)]
struct SampleFields {
    u32_val: u32,
    i32_val: i32,
    u64_val: u64,
    f32_val: f32,
    str_val: String,
}

impl Fields for SampleFields {
    fn name(&self) -> &str {
        "SampleFields"
    }

    fn visit_fields(&mut self, visitor: &mut dyn FieldsVisitor) {
        visitor.visit_u32(&mut self.u32_val);
        visitor.visit_i32(&mut self.i32_val);
        visitor.visit_u64(&mut self.u64_val);
        visitor.visit_f32(&mut self.f32_val);
        visitor.visit_string(&mut self.str_val);
    }
}

#[test]
fn fields_round_trip_and_golden() {
    let mut fields = SampleFields {
        u32_val: 1,
        i32_val: -2,
        u64_val: 0x1122_3344_5566_7788u64,
        f32_val: 1.5,
        str_val: "ab".to_string(),
    };

    let written = write_fields(&mut fields);

    let expected = vec![
        7,              // payload length in u32 (excluding this)
        1,              // u32_val
        0xFFFF_FFFEu32, // i32_val
        0x5566_7788u32, // u64 lower
        0x1122_3344u32, // u64 upper
        0x3FC0_0000u32, // f32 bits (1.5)
        1,              // string length in u32
        0x0000_6261u32, // 'a' 'b' \0 \0
    ];

    assert_eq!(written, expected);

    let mut read_back = SampleFields::default();
    let result = read_fields(&mut read_back, &written, 0);
    assert!(result.pos != 0);
    assert_eq!(read_back.u32_val, 1);
    assert_eq!(read_back.i32_val, -2);
    assert_eq!(read_back.u64_val, 0x1122_3344_5566_7788u64);
    assert_eq!(read_back.f32_val, 1.5);
    assert_eq!(read_back.str_val, "ab");
}
