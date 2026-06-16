//
// Created by jeason on 1/27/26.
//

#include "driver/cuda_macros.h"

#include <chrono>
#include <memory>
#include <thread>

#include <gtest/gtest.h>


namespace {
constexpr uint32_t kSleep = 5;
}

class MacrosTest : public ::testing::Test {
 public:
  void TearDown() { std::this_thread::sleep_for(std::chrono::milliseconds(kSleep)); }

 protected:
  bool ReturnTest(const std::shared_ptr<float> &ptr) {
    CUDA_RETURN_VAL_IF(ptr == nullptr, false);
    return true;
  }

  bool ReturnQuietTest(const std::shared_ptr<float> &ptr) {
    CUDA_RETURN_VAL_IF_QUIET(ptr == nullptr, false);
    return true;
  }

  bool CuCallTest(const int32_t device) {
    cudaDeviceProp device_prop;
    CUDART_RETURN_CALL(cudaGetDeviceProperties(&device_prop, device));
    CUDART_RETURN_CHECK();
    return true;
  }
};

TEST_F(MacrosTest, CudaLogTest) {
  std::shared_ptr<float> ptr = nullptr;
  EXPECT_FALSE(ReturnTest(ptr));
  EXPECT_FALSE(ReturnQuietTest(ptr));
  ptr = std::make_shared<float>(1);
  EXPECT_TRUE(ReturnTest(ptr));
  EXPECT_TRUE(ReturnQuietTest(ptr));

  CUDA_CHECK_NOTNULL(ptr);
  CUDA_CHECK(true);
}

TEST_F(MacrosTest, CudaRTLogTest) {
  cudaDeviceProp device_prop;
  CUDART_CALL(cudaGetDeviceProperties(&device_prop, -1));
  CUDART_CALL(cudaGetDeviceProperties(&device_prop, 0));
  CUDART_CHECK();

  EXPECT_FALSE(CuCallTest(-1));
  EXPECT_TRUE(CuCallTest(0));
}

