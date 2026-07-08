#include "im2col_rafa.cuh"

// im2col for ggml-native x(channel, t) input.
// Input:  x [C, T_in]  (ne[0]=C fastest, ne[1]=T) — ggml pipeline natural output.
//                           corresponds to PyTorch x [B, C, T] with T fastest.
// Output: dst [C*K, T_out]  (ne[0]=C*K, ne[1]=T_out)
// op_params: [K, s0, p0, d0]

template <typename T>
static __global__ void im2col_rafa_kernel(
        const float * __restrict__ x, T * __restrict__ dst,
        int C, int T_in, int T_out, int K, int s0, int p0, int d0) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int total = C * K * T_out;
    if (idx >= total) return;

    const int CK = C * K;
    // const int ic   = ic_k / K;
    // const int k    = ic_k % K;
    // idx = t * (C_in * K) + i * K + k
    const int ic_k = idx % CK; //i * K + k
    const int t = idx / CK;
    const int ic = ic_k / K;
    const int k = ic_k % K;

    const int inp = t * s0 - p0 + k * d0;
    if (inp >= 0 && inp < T_in)
        dst[ic_k + t * CK] = x[ic + inp * C];
    else
        dst[ic_k + t * CK] = T(0);
}

template <typename T>
static void im2col_rafa_cuda(const float * x, T * dst,
        int C, int T_in, int T_out, int K, int s0, int p0, int d0,
        cudaStream_t stream) {
    const int total = C * K * T_out;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;
    im2col_rafa_kernel<T><<<num_blocks, block_size, 0, stream>>>(
        x, dst, C, T_in, T_out, K, s0, p0, d0);
}

static void im2col_rafa_cuda_f16(const float * x, half * dst,
        int C, int T_in, int T_out, int K, int s0, int p0, int d0,
        cudaStream_t stream) {
    im2col_rafa_cuda<half>(x, dst, C, T_in, T_out, K, s0, p0, d0, stream);
}

static void im2col_rafa_cuda_f32(const float * x, float * dst,
        int C, int T_in, int T_out, int K, int s0, int p0, int d0,
        cudaStream_t stream) {
    im2col_rafa_cuda<float>(x, dst, C, T_in, T_out, K, s0, p0, d0, stream);
}

void ggml_cuda_op_im2col_rafa(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];  // [C, T_in]
    GGML_ASSERT(src->type == GGML_TYPE_F32);

    const int32_t * p = (const int32_t *)(dst->op_params);
    const int K   = p[0];
    const int s0  = p[1];
    const int p0  = p[2];
    const int d0  = p[3];

    const int C     = (int)src->ne[0];
    const int T_in  = (int)src->ne[1];
    const int T_out = (int)dst->ne[1];

    const float * x = (const float *)src->data;
    cudaStream_t stream = ctx.stream();

    if (dst->type == GGML_TYPE_F16) {
        im2col_rafa_cuda_f16(x, (half *)dst->data,
                             C, T_in, T_out, K, s0, p0, d0, stream);
    } else {
        im2col_rafa_cuda_f32(x, (float *)dst->data,
                             C, T_in, T_out, K, s0, p0, d0, stream);
    }
}
