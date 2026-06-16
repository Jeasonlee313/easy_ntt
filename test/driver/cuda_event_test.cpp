//
// Created by jeason on 5/9/26.
//

#include "driver/cuda_event.h"

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "driver/cuda_macros.h"
#include "driver/cuda_stream.h"

namespace driver {
namespace event {
namespace {
constexpr uint32_t kSleep = 5;
constexpr uint32_t kLargeNum = 1 << 20;
}  // namespace

class CudaEventTest : public ::testing::Test {
 public:
  void TearDown() { std::this_thread::sleep_for(std::chrono::milliseconds(kSleep)); }
};

TEST_F(CudaEventTest, DefaultConstruct) {
  CudaEvent event;
  EXPECT_NE(event.event(), nullptr);
  EXPECT_EQ(event.flags(), CudaEvent::Flag::DEFAULT_TIMING);
}

TEST_F(CudaEventTest, DisableTiming) {
  CudaEvent event(CudaEvent::Flag::DISABLE_TIMING);
  EXPECT_NE(event.event(), nullptr);
  EXPECT_EQ(event.flags(), CudaEvent::Flag::DISABLE_TIMING);

  stream::CudaStream cuda_stream(stream::CudaStream::Flag::NON_BLOCKING);
  EXPECT_TRUE(event.Record(cuda_stream));
  EXPECT_TRUE(event.Synchronize());
  EXPECT_TRUE(event.Query());
}

TEST_F(CudaEventTest, BlockingSync) {
  CudaEvent event(CudaEvent::Flag::BLOCKING_SYNC);
  EXPECT_NE(event.event(), nullptr);

  stream::CudaStream cuda_stream(stream::CudaStream::Flag::NON_BLOCKING);
  EXPECT_TRUE(event.Record(cuda_stream));
  EXPECT_TRUE(event.Synchronize());
  EXPECT_TRUE(event.Query());
}

TEST_F(CudaEventTest, DefaultFlag) {
  CudaEvent event(CudaEvent::Flag::DEFAULT);
  EXPECT_EQ(event.event(), nullptr);
  EXPECT_EQ(event.flags(), CudaEvent::Flag::DEFAULT);
  EXPECT_FALSE(event.Record(nullptr));
  EXPECT_FALSE(event.Synchronize());
  EXPECT_FALSE(event.Query());
}

TEST_F(CudaEventTest, RecordAndQuery) {
  stream::CudaStream cuda_stream(stream::CudaStream::Flag::NON_BLOCKING);
  CudaEvent event;

  float *d_data = nullptr;
  size_t byte_size = kLargeNum * sizeof(float);
  CUDART_CALL(cudaMalloc(reinterpret_cast<void **>(&d_data), byte_size));

  EXPECT_TRUE(event.Record(cuda_stream));
  CUDART_CALL(cudaMemsetAsync(d_data, 0, byte_size, cuda_stream.stream()));

  // Immediately after recording, the event may or may not be complete
  // Just verify Record succeeded
  EXPECT_TRUE(event.Synchronize());
  EXPECT_TRUE(event.Query());

  CUDART_CALL(cudaFree(d_data));
}

TEST_F(CudaEventTest, ElapsedTime) {
  stream::CudaStream cuda_stream(stream::CudaStream::Flag::NON_BLOCKING);
  CudaEvent start;
  CudaEvent end;

  float *d_data = nullptr;
  size_t byte_size = kLargeNum * sizeof(float);
  CUDART_CALL(cudaMalloc(reinterpret_cast<void **>(&d_data), byte_size));

  EXPECT_TRUE(start.Record(cuda_stream));
  CUDART_CALL(cudaMemsetAsync(d_data, 0, byte_size, cuda_stream.stream()));
  EXPECT_TRUE(end.Record(cuda_stream));
  EXPECT_TRUE(end.Synchronize());

  float ms = CudaEvent::ElapsedTime(start, end);
  EXPECT_GE(ms, 0.0f);

  CUDART_CALL(cudaFree(d_data));
}

TEST_F(CudaEventTest, Equality) {
  CudaEvent event1;
  CudaEvent event2 = event1;  // shared_ptr shared
  EXPECT_TRUE(event1 == event2);

  CudaEvent event3;
  EXPECT_FALSE(event1 == event3);
}

}  // namespace event
}  // namespace driver
