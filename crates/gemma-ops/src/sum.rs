//! Scalar sum helpers.

pub fn sum_f32(values: &[f32]) -> f32 {
    values.iter().copied().sum()
}

pub fn sum_bf16(values: &[half::bf16]) -> f32 {
    values.iter().map(|v| f32::from(*v)).sum()
}
