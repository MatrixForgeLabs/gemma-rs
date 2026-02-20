// On-device weight decompression kernels.
//
// These are placeholder stubs for Volta+ on-device decompression.
// Phase 2 initial implementation uses host-side decompression for all
// architectures (the Pascal path). These kernels will be completed when
// on-device decompression is prioritized.

// SFP decompression: one thread per element.
// SFP encoding: 1 byte per element, encoding a value in [-1.875, 1.875].
// Byte layout: sign(1) | exponent(3) | mantissa(4)
extern "C" __global__ void decompress_sfp(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        unsigned char byte = src[i];
        // SFP decoding: extract sign, exponent, mantissa
        int sign = (byte >> 7) & 1;
        int exp_bits = (byte >> 4) & 0x7;
        int mant_bits = byte & 0xF;

        float val;
        if (exp_bits == 0 && mant_bits == 0) {
            val = 0.0f;
        } else {
            // Reconstruct: value = (-1)^sign * 2^(exp-3) * (1 + mant/16)
            float mantissa = 1.0f + (float)mant_bits / 16.0f;
            float exponent = exp_bits - 3;
            val = ldexpf(mantissa, (int)exponent);
            if (sign) val = -val;
        }
        dst[i] = val;
    }
}

// Stub: NUQ and I8 decompression will be implemented when on-device
// decompression is needed (Volta+ path). For now, all compressed
// weights are decompressed on host before upload.
