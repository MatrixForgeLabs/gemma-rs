use gemma_ops::nn::{rms_norm, softmax};

#[test]
fn test_softmax() {
    let mut v = [1.0f32, 2.0, 3.0];
    softmax(&mut v);
    let sum: f32 = v.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);
}

#[test]
fn test_rms_norm() {
    let mut v = [1.0f32, 2.0, 3.0, 4.0];
    let scale = [1.0f32, 1.0, 1.0, 1.0];
    rms_norm(&mut v, &scale, 1e-6);
    let mut sum = 0.0f32;
    for x in v {
        sum += x * x;
    }
    let rms = (sum / 4.0).sqrt();
    // rms_norm multiplies by (1 + scale) after normalization, so with scale=1 the expected RMS is 2.0
    assert!((rms - 2.0).abs() < 1e-2, "rms_norm produced rms={}", rms);
}
