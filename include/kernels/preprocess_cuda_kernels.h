#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace gr00t {

bool LaunchNormalizePackU8C3ToBFloat16(const uint8_t* src, int src_height, int src_width, size_t src_pitch_bytes,
                                       float rescale_factor, float mean0, float mean1, float mean2, float std0,
                                       float std1, float std2, uint16_t* dst, cudaStream_t stream,
                                       std::string& error_message);

bool LaunchGatherBFloat16Embeddings(const int64_t* input_ids, const uint16_t* embedding_table, int64_t vocab_size,
                                    int64_t hidden_size, int64_t batch_size, int64_t seq_len, uint16_t* output,
                                    cudaStream_t stream, std::string& error_message);

bool LaunchFuseImageEmbeddings(const uint16_t* token_embeddings, const uint16_t* image_features,
                               const uint8_t* image_mask, int64_t batch_size, int64_t seq_len, int64_t hidden_size,
                               int64_t image_token_count, uint16_t* output, cudaStream_t stream,
                               std::string& error_message);

}  // namespace gr00t
