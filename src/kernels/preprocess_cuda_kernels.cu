#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <mutex>
#include <sstream>

#include "kernels/preprocess_cuda_kernels.h"

namespace gr00t {
namespace {

// Reuse tiny device flags across inferences to avoid cudaMalloc/cudaFree on the hot path
// (TokenIdsToEmbeddings + FuseImageEmbeddings).
int* DeviceIntScratch(std::once_flag& once, int** slot) {
  std::call_once(once, [slot]() {
    int* p = nullptr;
    if (cudaMalloc(&p, sizeof(int)) == cudaSuccess) {
      *slot = p;
    }
  });
  return *slot;
}

std::once_flag g_gather_err_flag_once;
int* g_gather_err_flag_device = nullptr;

std::once_flag g_fuse_overflow_flag_once;
int* g_fuse_overflow_flag_device = nullptr;

__device__ __forceinline__ uint16_t FloatToBFloat16BitsDevice(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t lsb = (bits >> 16) & 1U;
  const uint32_t rounding = 0x7FFFU + lsb;
  return static_cast<uint16_t>((bits + rounding) >> 16);
}

__global__ void NormalizePackU8C3ToBFloat16Kernel(const uint8_t* src, int src_height, int src_width,
                                                  size_t src_pitch_bytes, float rescale_factor, float mean0,
                                                  float mean1, float mean2, float inv_std0, float inv_std1,
                                                  float inv_std2, uint16_t* dst) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= src_width || y >= src_height) {
    return;
  }

  const uint8_t* row_ptr = src + static_cast<size_t>(y) * src_pitch_bytes;
  const int pixel_index = x * 3;
  const size_t hw_offset = static_cast<size_t>(y) * static_cast<size_t>(src_width) + static_cast<size_t>(x);
  const size_t channel_stride = static_cast<size_t>(src_height) * static_cast<size_t>(src_width);

  const float pixel0 = static_cast<float>(row_ptr[pixel_index + 0]);
  const float pixel1 = static_cast<float>(row_ptr[pixel_index + 1]);
  const float pixel2 = static_cast<float>(row_ptr[pixel_index + 2]);

  dst[hw_offset] = FloatToBFloat16BitsDevice((pixel0 * rescale_factor - mean0) * inv_std0);
  dst[channel_stride + hw_offset] = FloatToBFloat16BitsDevice((pixel1 * rescale_factor - mean1) * inv_std1);
  dst[2 * channel_stride + hw_offset] = FloatToBFloat16BitsDevice((pixel2 * rescale_factor - mean2) * inv_std2);
}

__global__ void GatherBFloat16EmbeddingsKernel(const int64_t* input_ids, const uint16_t* embedding_table,
                                               int64_t vocab_size, int64_t hidden_size, int64_t total_tokens,
                                               uint16_t* output, int* error_flag) {
  const int64_t token_index = static_cast<int64_t>(blockIdx.x);
  if (token_index >= total_tokens) {
    return;
  }

  const int64_t vocab_id = input_ids[token_index];
  if (vocab_id < 0 || vocab_id >= vocab_size) {
    if (threadIdx.x == 0) {
      atomicExch(error_flag, 1);
    }
    return;
  }

  const int64_t src_base = vocab_id * hidden_size;
  const int64_t dst_base = token_index * hidden_size;
  for (int64_t hidden_idx = threadIdx.x; hidden_idx < hidden_size; hidden_idx += blockDim.x) {
    output[dst_base + hidden_idx] = embedding_table[src_base + hidden_idx];
  }
}

// Image fusion: k-th masked token in sequence order uses image_features[:, k, :].
// Parallel path: exclusive prefix sum of mask per row -> deterministic image_idx (no atomics on idx).

__global__ void FuseImageEmbeddingsSerialKernel(const uint16_t* image_features, const uint8_t* image_mask,
                                                int64_t batch_size, int64_t seq_len, int64_t hidden_size,
                                                int64_t image_token_count, uint16_t* output, int* overflow_flag) {
  const int64_t batch_idx = static_cast<int64_t>(blockIdx.x);
  if (batch_idx >= batch_size || threadIdx.x != 0) {
    return;
  }

  int64_t image_idx = 0;
  for (int64_t t = 0; t < seq_len; ++t) {
    const int64_t token_index = batch_idx * seq_len + t;
    if (image_mask[token_index] == 0) {
      continue;
    }
    if (image_idx >= image_token_count) {
      atomicExch(overflow_flag, 1);
      return;
    }
    const int64_t src_base = (batch_idx * image_token_count + image_idx) * hidden_size;
    const int64_t dst_base = token_index * hidden_size;
    for (int64_t h = 0; h < hidden_size; ++h) {
      output[dst_base + h] = image_features[src_base + h];
    }
    ++image_idx;
  }
}

__global__ void FuseImageEmbeddingsParallelKernel(const uint16_t* image_features, const uint8_t* image_mask,
                                                  int batch_size, int seq_len, int64_t hidden_size,
                                                  int64_t image_token_count, uint16_t* output, int* overflow_flag) {
  extern __shared__ int s_excl[];
  __shared__ int s_ok;

  const int b = blockIdx.x;
  if (b >= batch_size) {
    return;
  }

  const int row_base = b * seq_len;

  if (threadIdx.x == 0) {
    int run = 0;
    for (int t = 0; t < seq_len; ++t) {
      s_excl[t] = run;
      run += (image_mask[row_base + t] != 0) ? 1 : 0;
    }
    const int need = run;
    s_ok = (need <= static_cast<int>(image_token_count)) ? 1 : 0;
    if (!s_ok) {
      atomicExch(overflow_flag, 1);
    }
  }
  __syncthreads();
  if (!s_ok) {
    return;
  }

  for (int t = 0; t < seq_len; ++t) {
    if (image_mask[row_base + t] == 0) {
      continue;
    }
    const int image_idx = s_excl[t];
    const int64_t src_base =
        (static_cast<int64_t>(b) * image_token_count + static_cast<int64_t>(image_idx)) * hidden_size;
    const int64_t dst_base = (static_cast<int64_t>(row_base) + static_cast<int64_t>(t)) * hidden_size;
    for (int64_t h = threadIdx.x; h < hidden_size; h += blockDim.x) {
      output[dst_base + h] = image_features[src_base + h];
    }
    __syncthreads();
  }
}

}  // namespace

bool LaunchNormalizePackU8C3ToBFloat16(const uint8_t* src, int src_height, int src_width, size_t src_pitch_bytes,
                                       float rescale_factor, float mean0, float mean1, float mean2, float std0,
                                       float std1, float std2, uint16_t* dst, cudaStream_t stream,
                                       std::string& error_message) {
  if (src == nullptr || dst == nullptr || src_height <= 0 || src_width <= 0) {
    error_message = "invalid input for NormalizePackU8C3ToBFloat16";
    return false;
  }

  const auto safe_inv_std = [](float std_value) {
    const float denom = std::abs(std_value) > std::numeric_limits<float>::epsilon() ? std_value : 1.0f;
    return 1.0f / denom;
  };

  const dim3 block_dim(16, 16);
  const dim3 grid_dim((static_cast<unsigned int>(src_width) + block_dim.x - 1) / block_dim.x,
                      (static_cast<unsigned int>(src_height) + block_dim.y - 1) / block_dim.y);

  NormalizePackU8C3ToBFloat16Kernel<<<grid_dim, block_dim, 0, stream>>>(
      src, src_height, src_width, src_pitch_bytes, rescale_factor, mean0, mean1, mean2, safe_inv_std(std0),
      safe_inv_std(std1), safe_inv_std(std2), dst);

  const cudaError_t launch_error = cudaGetLastError();
  if (launch_error != cudaSuccess) {
    std::ostringstream os;
    os << "NormalizePackU8C3ToBFloat16Kernel launch failed: " << cudaGetErrorString(launch_error);
    error_message = os.str();
    return false;
  }
  return true;
}

bool LaunchGatherBFloat16Embeddings(const int64_t* input_ids, const uint16_t* embedding_table, int64_t vocab_size,
                                    int64_t hidden_size, int64_t batch_size, int64_t seq_len, uint16_t* output,
                                    cudaStream_t stream, std::string& error_message) {
  if (input_ids == nullptr || embedding_table == nullptr || output == nullptr || vocab_size <= 0 || hidden_size <= 0 ||
      batch_size <= 0 || seq_len <= 0) {
    error_message = "invalid input for GatherBFloat16Embeddings";
    return false;
  }

  const int64_t total_tokens = batch_size * seq_len;
  int* d_error_flag = DeviceIntScratch(g_gather_err_flag_once, &g_gather_err_flag_device);
  if (d_error_flag == nullptr) {
    error_message = "GatherBFloat16Embeddings device scratch alloc failed";
    return false;
  }
  cudaError_t err = cudaMemsetAsync(d_error_flag, 0, sizeof(int), stream);
  if (err != cudaSuccess) {
    error_message = std::string("cudaMemsetAsync error_flag failed: ") + cudaGetErrorString(err);
    return false;
  }

  constexpr int kThreads = 256;
  GatherBFloat16EmbeddingsKernel<<<static_cast<unsigned int>(total_tokens), kThreads, 0, stream>>>(
      input_ids, embedding_table, vocab_size, hidden_size, total_tokens, output, d_error_flag);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    error_message = std::string("GatherBFloat16EmbeddingsKernel launch failed: ") + cudaGetErrorString(err);
    return false;
  }

  int h_error_flag = 0;
  err = cudaMemcpyAsync(&h_error_flag, d_error_flag, sizeof(int), cudaMemcpyDeviceToHost, stream);
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
  }
  if (err != cudaSuccess) {
    error_message = std::string("GatherBFloat16Embeddings error check failed: ") + cudaGetErrorString(err);
    return false;
  }
  if (h_error_flag != 0) {
    error_message = "GatherBFloat16Embeddings found out-of-range token id";
    return false;
  }
  return true;
}

bool LaunchFuseImageEmbeddings(const uint16_t* token_embeddings, const uint16_t* image_features,
                               const uint8_t* image_mask, int64_t batch_size, int64_t seq_len, int64_t hidden_size,
                               int64_t image_token_count, uint16_t* output, cudaStream_t stream,
                               std::string& error_message) {
  if (token_embeddings == nullptr || image_features == nullptr || image_mask == nullptr || output == nullptr ||
      batch_size <= 0 || seq_len <= 0 || hidden_size <= 0 || image_token_count <= 0) {
    error_message = "invalid input for FuseImageEmbeddings";
    return false;
  }

  const size_t total_bytes = static_cast<size_t>(batch_size * seq_len * hidden_size) * sizeof(uint16_t);
  cudaError_t err = cudaMemcpyAsync(output, token_embeddings, total_bytes, cudaMemcpyDeviceToDevice, stream);
  if (err != cudaSuccess) {
    error_message = std::string("cudaMemcpyAsync token->output failed: ") + cudaGetErrorString(err);
    return false;
  }

  int* d_overflow_flag = DeviceIntScratch(g_fuse_overflow_flag_once, &g_fuse_overflow_flag_device);
  if (d_overflow_flag == nullptr) {
    error_message = "FuseImageEmbeddings device scratch alloc failed";
    return false;
  }
  err = cudaMemsetAsync(d_overflow_flag, 0, sizeof(int), stream);
  if (err != cudaSuccess) {
    error_message = std::string("cudaMemsetAsync d_overflow_flag failed: ") + cudaGetErrorString(err);
    return false;
  }

  constexpr size_t kMaxParallelPrefixBytes = 40 * 1024;
  const bool seq_fits_int = (seq_len > 0) && (seq_len == static_cast<int64_t>(static_cast<int>(seq_len)));
  const int seq_len_i = static_cast<int>(seq_len);
  const size_t prefix_shmem = static_cast<size_t>(seq_len_i) * sizeof(int);
  const bool use_parallel = seq_fits_int && prefix_shmem > 0 && prefix_shmem <= kMaxParallelPrefixBytes &&
                            batch_size <= static_cast<int64_t>(std::numeric_limits<int>::max());

  if (use_parallel) {
    constexpr unsigned int kFuseThreads = 256;
    FuseImageEmbeddingsParallelKernel<<<static_cast<unsigned int>(batch_size), kFuseThreads,
                                        static_cast<unsigned int>(prefix_shmem), stream>>>(
        image_features, image_mask, static_cast<int>(batch_size), seq_len_i, hidden_size, image_token_count, output,
        d_overflow_flag);
  } else {
    FuseImageEmbeddingsSerialKernel<<<static_cast<unsigned int>(batch_size), 1, 0, stream>>>(
        image_features, image_mask, batch_size, seq_len, hidden_size, image_token_count, output, d_overflow_flag);
  }

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    error_message = std::string(use_parallel ? "FuseImageEmbeddingsParallelKernel launch failed: "
                                             : "FuseImageEmbeddingsSerialKernel launch failed: ") +
                    cudaGetErrorString(err);
    return false;
  }

  int h_overflow_flag = 0;
  err = cudaMemcpyAsync(&h_overflow_flag, d_overflow_flag, sizeof(int), cudaMemcpyDeviceToHost, stream);
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
  }
  if (err != cudaSuccess) {
    error_message = std::string("FuseImageEmbeddings overflow check failed: ") + cudaGetErrorString(err);
    return false;
  }
  if (h_overflow_flag != 0) {
    error_message = "FuseImageEmbeddings image feature count is smaller than image mask token count";
    return false;
  }
  return true;
}

}  // namespace gr00t
