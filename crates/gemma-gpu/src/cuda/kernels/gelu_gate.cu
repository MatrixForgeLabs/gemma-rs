// Fused GELU-gated linear unit.
//
// src has 2*hidden_dim elements: gate (first half), up (second half).
// dst[i] = gelu(gate[i]) * up[i]
//
// GELU approximation: 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x^3)))

extern "C" __global__ void gelu_gate(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int hidden_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < hidden_dim; i += stride) {
        float g = src[i];
        float u = src[i + hidden_dim];

        // GELU approximation
        float g3 = g * g * g;
        float inner = 0.7978845608f * (g + 0.044715f * g3);
        float gelu_val = 0.5f * g * (1.0f + tanhf(inner));

        dst[i] = gelu_val * u;
    }
}
