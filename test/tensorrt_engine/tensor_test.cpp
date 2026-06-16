//
// Created by jeason on 1/28/26.
//

#include "tensorrt_engine/tensor.h"

#include <chrono>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

namespace tensorrt_engine {
namespace tensor {
namespace {
constexpr uint32_t kSleep = 5;
constexpr uint32_t kDefaultNum = 1 << 15;
constexpr uint32_t kLargeNum = 1 << 25;
constexpr uint32_t kFloatMax = 1e4;
}  // namespace
class TensorTest : public ::testing::Test {
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

TEST_F(TensorTest, FloatTest) {
  Tensor d_empty;
  EXPECT_TRUE(d_empty.data() == nullptr);
  EXPECT_TRUE(d_empty.mutable_data() == nullptr);
  Tensor d_data;
  nvinfer1::Dims dims;
  dims.nbDims = 1;
  dims.d[0] = kDefaultNum;
  EXPECT_TRUE(d_data.Resize(dims, nvinfer1::DataType::kFLOAT));
  EXPECT_TRUE(d_data.data() != nullptr);
  EXPECT_TRUE(d_data.mutable_data() != nullptr);

  std::vector<float> dst_h_data(kDefaultNum);

  EXPECT_TRUE(memcmp(src_h_data, dst_h_data.data(), kDefaultNum * sizeof(float)) != 0);
  EXPECT_TRUE(d_data.CopyFromHost(src_h_data));
  EXPECT_TRUE(d_data.CopyToHost(dst_h_data.data()));
  EXPECT_EQ(memcmp(src_h_data, dst_h_data.data(), kDefaultNum * sizeof(float)), 0);
}

}  // namespace tensor
}  // namespace tensorrt_engine
