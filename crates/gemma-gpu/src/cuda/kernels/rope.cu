// Rotary Position Embedding (RoPE) in-place.
//
// For dimension pair (dim, dim + half):
//   freq_exp = 2.0 * dim / rope_dim
//   inv_timescale = 1.0 / 10000^freq_exp
//   theta = pos * inv_timescale
//   x0_new = x0 * cos(theta) - x1 * sin(theta)
//   x1_new = x0 * sin(theta) + x1 * cos(theta)
//
// Grid-stride loop over half the dimensions.

extern "C" __global__ void rope_inplace(
    float* __restrict__ values,
    int rope_dim,
    float pos
) {
    int half = rope_dim / 2;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int dim = tid; dim < half; dim += stride) {
        float freq_exp = (2.0f * (float)dim) / (float)rope_dim;
        float inv_timescale = 1.0f / powf(10000.0f, freq_exp);
        float theta = pos * inv_timescale;
        float cos_t = cosf(theta);
        float sin_t = sinf(theta);

        float x0 = values[dim];
        float x1 = values[dim + half];
        values[dim]        = x0 * cos_t - x1 * sin_t;
        values[dim + half] = x0 * sin_t + x1 * cos_t;
    }
}
