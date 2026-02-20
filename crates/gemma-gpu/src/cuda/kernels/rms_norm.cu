// RMS Normalization: values[i] = values[i] * inv_rms * (1.0 + scale[i])
// where inv_rms = rsqrt(mean(values^2) + eps)
//
// Single-block kernel: uses shared memory reduction for the RMS computation,
// then elementwise scale. Suited for vectors up to ~8192 elements (model_dim).

extern "C" __global__ void rms_norm(
    float* __restrict__ values,
    const float* __restrict__ scale,
    int n,
    float eps
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Phase 1: compute sum of squares via parallel reduction.
    float local_sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        float v = values[i];
        local_sum += v * v;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Tree reduction in shared memory.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Phase 2: compute inverse RMS and apply elementwise.
    float inv_rms = rsqrtf(sdata[0] / (float)n + eps);
    __syncthreads();

    for (int i = tid; i < n; i += stride) {
        values[i] = values[i] * inv_rms * (1.0f + scale[i]);
    }
}
