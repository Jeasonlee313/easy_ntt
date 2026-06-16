//
// Created by jeason on 5/9/26.
//

#include "driver/cuda_graph.h"

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "driver/cuda_macros.h"
#include "driver/cuda_stream.h"

namespace driver {
namespace graph {
namespace {
constexpr uint32_t kSleep = 5;
constexpr size_t kDataSize = 1024;
}  // namespace

class CudaGraphTest : public ::testing::Test {
 public:
  void TearDown() { std::this_thread::sleep_for(std::chrono::milliseconds(kSleep)); }
};

TEST_F(CudaGraphTest, ConstructAndDestroy) {
  stream::CudaStream cuda_stream(stream::CudaStream::Flag::NON_BLOCKING);
  { CudaGraph graph(cuda_stream); }
  // Should not crash on destruction
}

TEST_F(CudaGraphTest, CaptureAndLaunch) {
  stream::CudaStream cuda_stream(stream::CudaStream::Flag::NON_BLOCKING);
  CudaGraph graph(cuda_stream);

  float *d_data = nullptr;
  size_t byte_size = kDataSize * sizeof(float);
  CUDART_CALL(cudaMalloc(reinterpret_cast<void **>(&d_data), byte_size));

  EXPECT_FALSE(graph.IsCapturing());
  EXPECT_TRUE(graph.BeginCapture());
  EXPECT_TRUE(graph.IsCapturing());

  // Submit operations to the capture stream
  CUDART_CALL(cudaMemsetAsync(d_data, 0, byte_size, cuda_stream.stream()));

  EXPECT_TRUE(graph.EndCapture());
  EXPECT_FALSE(graph.IsCapturing());

  // Instantiate and launch
  CudaGraphExec exec = graph.Instantiate();
  EXPECT_NE(exec.exec(), nullptr);

  EXPECT_TRUE(exec.Launch(cuda_stream));
  EXPECT_TRUE(cuda_stream.Synchronize());

  CUDART_CALL(cudaFree(d_data));
}

TEST_F(CudaGraphTest, EmptyExecLaunch) {
  CudaGraphExec exec;
  EXPECT_EQ(exec.exec(), nullptr);

  stream::CudaStream cuda_stream(stream::CudaStream::Flag::NON_BLOCKING);
  EXPECT_FALSE(exec.Launch(cuda_stream));
}

TEST_F(CudaGraphTest, ReuseGraphCapture) {
  stream::CudaStream cuda_stream(stream::CudaStream::Flag::NON_BLOCKING);
  CudaGraph graph(cuda_stream);

  float *d_data = nullptr;
  CUDART_CALL(cudaMalloc(reinterpret_cast<void **>(&d_data), kDataSize * sizeof(float)));

  // First capture
  EXPECT_TRUE(graph.BeginCapture());
  CUDART_CALL(cudaMemsetAsync(d_data, 0, kDataSize * sizeof(float), cuda_stream.stream()));
  EXPECT_TRUE(graph.EndCapture());

  auto exec1 = graph.Instantiate();
  EXPECT_NE(exec1.exec(), nullptr);

  // Second capture (reuse the same graph object)
  EXPECT_TRUE(graph.BeginCapture());
  CUDART_CALL(cudaMemsetAsync(d_data, 1, kDataSize * sizeof(float), cuda_stream.stream()));
  EXPECT_TRUE(graph.EndCapture());

  auto exec2 = graph.Instantiate();
  EXPECT_NE(exec2.exec(), nullptr);

  EXPECT_TRUE(exec2.Launch(cuda_stream));
  EXPECT_TRUE(cuda_stream.Synchronize());

  CUDART_CALL(cudaFree(d_data));
}

TEST_F(CudaGraphTest, ExecEquality) {
  CudaGraphExec exec1;
  CudaGraphExec exec2 = exec1;
  EXPECT_TRUE(exec1 == exec2);
}

}  // namespace graph
}  // namespace driver
