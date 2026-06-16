//
// Created by jeason on 1/28/26.
//

#include "driver/cuda_base_memory.h"

#include <chrono>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include "driver/cuda_macros.h"
#include "driver/cuda_memory_type.h"

namespace driver {
namespace base_memory {
namespace {
constexpr uint32_t kSleep = 5;
constexpr uint32_t kDefaultNum = 1 << 15;
constexpr uint32_t kLargeNum = 1 << 25;
constexpr uint32_t kFloatMax = 1e4;
}  // namespace
class CudaBaseMemoryTest : public ::testing::Test {
 public:
  void SetUp() override {
    src_h_data = static_cast<float *>(malloc(kDefaultNum * sizeof(float)));
    for (size_t i = 0; i < kDefaultNum; ++i) {
      *(src_h_data + i) =
          static_cast<float>(rand_r(&seed_)) / static_cast<float>(RAND_MAX / (kFloatMax << 1)) - kFloatMax;
    }
  }

  void TearDown() override {
    // Wait for the log to be printed asynchronously
    std::this_thread::sleep_for(std::chrono::milliseconds(kSleep));
    free(src_h_data);
  }

  uint32_t seed_ = 123;
  float *src_h_data = nullptr;
};

TEST_F(CudaBaseMemoryTest, DeviceTest) {
  CudaBaseMemory d_empty(CudaMemoryType::DEVICE);
  EXPECT_EQ(d_empty.byte_size(), 0);
  EXPECT_TRUE(d_empty.ptr() == nullptr);
  EXPECT_TRUE(d_empty.type() == CudaMemoryType::DEVICE);
  CudaBaseMemory d_data(CudaMemoryType::DEVICE);
  EXPECT_TRUE(d_data.AllocateMemory(kDefaultNum * sizeof(float)));
  EXPECT_EQ(d_data.byte_size(), kDefaultNum * sizeof(float));
  EXPECT_TRUE(d_data.ptr() != nullptr);
  EXPECT_TRUE(d_data.type() == CudaMemoryType::DEVICE);

  CudaBaseMemory dst_h_data(CudaMemoryType::PINNED);
  EXPECT_TRUE(dst_h_data.AllocateMemory(kDefaultNum * sizeof(float)));
  EXPECT_TRUE(dst_h_data.Memset(0, 0, dst_h_data.byte_size()));

  EXPECT_TRUE(d_data.MemcpyFromHost(src_h_data, 0, kDefaultNum * sizeof(float)));
  EXPECT_TRUE(d_data.MemcpyToHost(0, kDefaultNum * sizeof(float), dst_h_data.ptr()));
  EXPECT_EQ(memcmp(src_h_data, dst_h_data.ptr(), kDefaultNum * sizeof(float)), 0);
}

TEST_F(CudaBaseMemoryTest, AllocateMemoryAsync) {
  CudaBaseMemory d_data(CudaMemoryType::DEVICE);
  d_data.SetStream(std::make_shared<stream::CudaStream>(stream::CudaStream::Flag::NON_BLOCKING));
  EXPECT_TRUE(d_data.AllocateMemoryAsync(kDefaultNum * sizeof(float)));
  EXPECT_EQ(d_data.byte_size(), kDefaultNum * sizeof(float));
  EXPECT_TRUE(d_data.ptr() != nullptr);

  // Verify data can be written and read back
  EXPECT_TRUE(d_data.MemcpyFromHost(src_h_data, 0, kDefaultNum * sizeof(float)));

  CudaBaseMemory dst_h_data(CudaMemoryType::PINNED);
  EXPECT_TRUE(dst_h_data.AllocateMemory(kDefaultNum * sizeof(float)));
  EXPECT_TRUE(d_data.MemcpyToHost(0, kDefaultNum * sizeof(float), dst_h_data.ptr()));
  EXPECT_EQ(memcmp(src_h_data, dst_h_data.ptr(), kDefaultNum * sizeof(float)), 0);
}

TEST_F(CudaBaseMemoryTest, RegisterHostMemory) {
  float *external_buffer = static_cast<float *>(malloc(kDefaultNum * sizeof(float)));
  memcpy(external_buffer, src_h_data, kDefaultNum * sizeof(float));

  {
    CudaBaseMemory mem(CudaMemoryType::PINNED);
    EXPECT_TRUE(mem.RegisterHostMemory(external_buffer, kDefaultNum * sizeof(float)));
    EXPECT_EQ(mem.byte_size(), kDefaultNum * sizeof(float));
    EXPECT_TRUE(mem.ptr() != nullptr);
    EXPECT_EQ(mem.ptr(), external_buffer);

    // Verify async H2D from registered memory works
    CudaBaseMemory d_data(CudaMemoryType::DEVICE);
    EXPECT_TRUE(d_data.AllocateMemory(kDefaultNum * sizeof(float)));
    EXPECT_TRUE(d_data.MemcpyFromHost(external_buffer, 0, kDefaultNum * sizeof(float)));
  }
  // mem destructed, cudaHostUnregister should have been called

  free(external_buffer);
}

TEST_F(CudaBaseMemoryTest, RegisterHostMemoryInvalidType) {
  float buffer[16] = {0};
  CudaBaseMemory mem(CudaMemoryType::DEVICE);
  EXPECT_FALSE(mem.RegisterHostMemory(buffer, sizeof(buffer)));
}

TEST_F(CudaBaseMemoryTest, PinnedMemoryWriteCombined) {
  CudaBaseMemory mem(CudaMemoryType::PINNED_WC);
  EXPECT_TRUE(mem.AllocateMemory(kDefaultNum * sizeof(float)));
  EXPECT_TRUE(mem.ptr() != nullptr);
  EXPECT_EQ(mem.byte_size(), kDefaultNum * sizeof(float));

  // Verify data can be written
  memcpy(mem.ptr(), src_h_data, kDefaultNum * sizeof(float));

  // Async H2D from WriteCombined memory
  CudaBaseMemory d_data(CudaMemoryType::DEVICE);
  EXPECT_TRUE(d_data.AllocateMemory(kDefaultNum * sizeof(float)));
  EXPECT_TRUE(d_data.MemcpyFromHost(mem.ptr(), 0, kDefaultNum * sizeof(float)));
}

}  // namespace base_memory
}  // namespace driver
