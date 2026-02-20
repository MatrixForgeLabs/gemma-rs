//! Basic neural network ops (scalar baselines).

pub fn softmax(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    let mut max = values[0];
    for &v in values.iter().skip(1) {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in values.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum != 0.0 {
        for v in values.iter_mut() {
            *v /= sum;
        }
    }
}

pub fn rms_norm(values: &mut [f32], scale: &[f32], eps: f32) {
    assert_eq!(values.len(), scale.len());
    let mut sum = 0.0f32;
    for &v in values.iter() {
        sum += v * v;
    }
    let inv = (sum / values.len() as f32 + eps).sqrt().recip();
    for i in 0..values.len() {
        values[i] = values[i] * inv * (1.0 + scale[i]);
    }
}

pub fn rope_inplace(values: &mut [f32], pos: f32) {
    assert!(values.len() % 2 == 0);
    let rope_dim = values.len();
    let half = rope_dim / 2;
    for dim in 0..half {
        let freq_exp = (2.0 * dim as f32) / rope_dim as f32;
        let inv_timescale = 1.0 / 10000.0f32.powf(freq_exp);
        let theta = pos * inv_timescale;
        let cos = theta.cos();
        let sin = theta.sin();
        let x0 = values[dim];
        let x1 = values[dim + half];
        values[dim] = x0 * cos - x1 * sin;
        values[dim + half] = x0 * sin + x1 * cos;
    }
}
