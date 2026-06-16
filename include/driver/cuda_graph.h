//
// Created by jeason on 5/9/26.
//

#ifndef TENSORRT_ENGINE_CUDA_GRAPH_H
#define TENSORRT_ENGINE_CUDA_GRAPH_H

#include <cuda_runtime_api.h>

#include <memory>

#include "driver/cuda_stream.h"

namespace driver {
namespace graph {

// Forward declaration
class CudaGraphExec;

class CudaGraph {
 public:
  explicit CudaGraph(const stream::CudaStream &capture_stream);
  ~CudaGraph();

  // Disable copy, allow move
  CudaGraph(const CudaGraph &) = delete;
  CudaGraph &operator=(const CudaGraph &) = delete;
  CudaGraph(CudaGraph &&) noexcept = default;
  CudaGraph &operator=(CudaGraph &&) noexcept = default;

  // Capture control
  [[nodiscard]] bool BeginCapture(cudaStreamCaptureMode mode = cudaStreamCaptureModeGlobal);
  [[nodiscard]] bool EndCapture();
  [[nodiscard]] bool IsCapturing() const;

  // Access the bound capture stream
  [[nodiscard]] const stream::CudaStream &capture_stream() const;

  // Instantiate into an executable graph
  [[nodiscard]] CudaGraphExec Instantiate() const;

  [[nodiscard]] cudaGraph_t graph() const;

 protected:
  std::shared_ptr<cudaGraph_t> graph_ = nullptr;
  std::shared_ptr<stream::CudaStream> capture_stream_ = nullptr;
  bool is_capturing_ = false;
};

class CudaGraphExec {
 public:
  friend class CudaGraph;

  CudaGraphExec();
  ~CudaGraphExec() = default;

  // Exec instances can be shared (similar to CudaStream)
  CudaGraphExec(const CudaGraphExec &) = default;
  CudaGraphExec &operator=(const CudaGraphExec &) = default;
  CudaGraphExec(CudaGraphExec &&) noexcept = default;
  CudaGraphExec &operator=(CudaGraphExec &&) noexcept = default;

  // Launch on the specified stream
  [[nodiscard]] bool Launch(cudaStream_t stream) const;
  [[nodiscard]] bool Launch(const stream::CudaStream &cuda_stream) const;

  // Update parameters from a new graph definition (CUDA 12+)
  [[nodiscard]] bool Update(const CudaGraph &graph);

  [[nodiscard]] cudaGraphExec_t exec() const;

  bool operator==(const CudaGraphExec &other) const;

 protected:
  std::shared_ptr<cudaGraphExec_t> exec_ = nullptr;
};

}  // namespace graph
}  // namespace driver

#endif  // TENSORRT_ENGINE_CUDA_GRAPH_H
