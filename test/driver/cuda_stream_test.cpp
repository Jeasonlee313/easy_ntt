//
// Created by jeason on 1/28/26.
//
#include "driver/cuda_stream.h"

#include <chrono>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include "driver/cuda_event.h"
#include "driver/cuda_macros.h"

namespace driver {
namespace stream {
namespace {
constexpr uint32_t kSleep = 5;
constexpr uint32_t kLargeNum = 1 << 25;
constexpr uint32_t kFloatMax = 1e4;
}  // namespace

class CudaStreamTest : public ::testing::Test {
 public:
  void TearDown() { std::this_thread::sleep_for(std::chrono::milliseconds(kSleep)); }

 protected:
  uint32_t seed_ = 123;
};

TEST_F(CudaStreamTest, DefaultConstructor) {
  CudaStream stream;
  EXPECT_NE(stream.stream(), nullptr);
  EXPECT_TRUE(stream.Synchronize());
}

TEST_F(CudaStreamTest, AsyncStream) {
  CudaStream stream(CudaStream::Flag::NON_BLOCKING);
  float *d_data = nullptr;
  float *h_src_data = nullptr;
  float *h_dst_data = nullptr;

  size_t byte_size = kLargeNum * sizeof(float);

  CUDART_CALL(cudaMalloc(reinterpret_cast<void **>(&d_data), byte_size));
  CUDART_CALL(cudaMallocHost(reinterpret_cast<void **>(&h_src_data), byte_size));
  CUDART_CALL(cudaMallocHost(reinterpret_cast<void **>(&h_dst_data), byte_size));

  for (size_t i = 0; i < kLargeNum; ++i) {
    *(h_src_data + i) =
        static_cast<float>(rand_r(&seed_)) / static_cast<float>(RAND_MAX / (kFloatMax << 1)) - kFloatMax;
  }

  CUDART_CALL(cudaMemcpyAsync(d_data, h_src_data, byte_size, cudaMemcpyHostToDevice, stream.stream()));
  CUDART_CALL(cudaMemcpyAsync(h_dst_data, d_data, byte_size, cudaMemcpyDeviceToHost, stream.stream()));

  // EXPECT_NE(memcmp(h_src_data, h_dst_data, byte_size), 0);
  EXPECT_TRUE(stream.Synchronize());
  EXPECT_EQ(memcmp(h_src_data, h_dst_data, byte_size), 0);
  CUDART_CALL(cudaFree(d_data));
  CUDART_CALL(cudaFreeHost(h_src_data));
  CUDART_CALL(cudaFreeHost(h_dst_data));
}

TEST_F(CudaStreamTest, Operator) {
  CudaStream stream1(CudaStream::Flag::NON_BLOCKING);
  EXPECT_TRUE(stream1 == stream1);
  CudaStream stream(CudaStream::Flag::BLOCKING);
  EXPECT_FALSE(stream == stream1);
}

TEST_F(CudaStreamTest, WaitEvent) {
  CudaStream stream_a(CudaStream::Flag::NON_BLOCKING);
  CudaStream stream_b(CudaStream::Flag::NON_BLOCKING);
  event::CudaEvent evt;

  float *d_data = nullptr;
  CUDART_CALL(cudaMalloc(reinterpret_cast<void **>(&d_data), 1024 * sizeof(float)));

  // Record event on stream_a after some work
  CUDART_CALL(cudaMemsetAsync(d_data, 0, 1024 * sizeof(float), stream_a.stream()));
  EXPECT_TRUE(evt.Record(stream_a));

  // stream_b waits for the event
  EXPECT_TRUE(stream_b.WaitEvent(evt));

  // stream_b can now safely use d_data
  CUDART_CALL(cudaMemsetAsync(d_data, 1, 1024 * sizeof(float), stream_b.stream()));
  EXPECT_TRUE(stream_b.Synchronize());

  CUDART_CALL(cudaFree(d_data));
}

TEST_F(CudaStreamTest, HighPriority) {
  CudaStream high_stream(CudaStream::Flag::HIGH_PRIORITY);
  EXPECT_NE(high_stream.stream(), nullptr);
  EXPECT_TRUE(high_stream.Synchronize());
}

TEST_F(CudaStreamTest, LowPriority) {
  CudaStream low_stream(CudaStream::Flag::LOW_PRIORITY);
  EXPECT_NE(low_stream.stream(), nullptr);
  EXPECT_TRUE(low_stream.Synchronize());
}

TEST_F(CudaStreamTest, ExplicitPriority) {
  int least_priority = 0;
  int greatest_priority = 0;
  CUDART_CALL(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
  CudaStream stream(greatest_priority);
  EXPECT_NE(stream.stream(), nullptr);
  EXPECT_TRUE(stream.Synchronize());
}

}  // namespace stream
}  // namespace driver
