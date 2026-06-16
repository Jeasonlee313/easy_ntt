//
// Created by jeason on 5/9/26.
//

#include "driver/cuda_graph.h"

#include "driver/cuda_macros.h"

namespace driver {
namespace graph {

// ==================== CudaGraph ====================

CudaGraph::CudaGraph(const stream::CudaStream &capture_stream)
    : capture_stream_(std::make_shared<stream::CudaStream>(capture_stream)), is_capturing_(false) {
  auto raw_graph = new cudaGraph_t;
  CUDART_RETURN_VAL_IF_FAIL(cudaGraphCreate(raw_graph, 0), );
  graph_ = std::shared_ptr<cudaGraph_t>(raw_graph, [](cudaGraph_t *const g) {
    if (g != nullptr && *g != nullptr) {
      cudaGraphDestroy(*g);
    }
    delete g;
  });
}

CudaGraph::~CudaGraph() {
  if (is_capturing_ && capture_stream_ != nullptr) {
    cudaGraph_t raw_graph = nullptr;
    cudaError_t err = cudaStreamEndCapture(capture_stream_->stream(), &raw_graph);
    if (err != cudaSuccess) {
      LOG(ERROR) << "CUDA API Error in ~CudaGraph: " << cudaGetErrorString(err);
      cudaGetLastError();
    } else if (raw_graph != nullptr) {
      cudaGraphDestroy(raw_graph);
    }
    is_capturing_ = false;
  }
}

bool CudaGraph::BeginCapture(cudaStreamCaptureMode mode) {
  CUDA_RETURN_VAL_IF(is_capturing_, false);
  CUDA_RETURN_VAL_IF(capture_stream_ == nullptr, false);
  CUDA_RETURN_VAL_IF(graph_ == nullptr, false);
  CUDART_RETURN_CALL(cudaStreamBeginCapture(capture_stream_->stream(), mode));
  is_capturing_ = true;
  return true;
}

bool CudaGraph::EndCapture() {
  CUDA_RETURN_VAL_IF(!is_capturing_, false);
  CUDA_RETURN_VAL_IF(capture_stream_ == nullptr, false);

  cudaGraph_t raw_graph = nullptr;
  CUDART_RETURN_CALL(cudaStreamEndCapture(capture_stream_->stream(), &raw_graph));

  if (raw_graph != nullptr) {
    auto new_graph = new cudaGraph_t(raw_graph);
    graph_ = std::shared_ptr<cudaGraph_t>(new_graph, [](cudaGraph_t *const g) {
      if (g != nullptr && *g != nullptr) {
        cudaGraphDestroy(*g);
      }
      delete g;
    });
  } else {
    LOG(WARNING) << "cudaStreamEndCapture returned nullptr graph";
  }

  is_capturing_ = false;
  return true;
}

bool CudaGraph::IsCapturing() const { return is_capturing_; }

const stream::CudaStream &CudaGraph::capture_stream() const {
  CUDA_CHECK(capture_stream_ != nullptr);
  return *capture_stream_;
}

CudaGraphExec CudaGraph::Instantiate() const {
  CudaGraphExec exec;
  if (graph_ == nullptr || *graph_ == nullptr) {
    return exec;
  }

  auto raw_exec = new cudaGraphExec_t;
  CUDART_RETURN_VAL_IF_FAIL(cudaGraphInstantiate(raw_exec, *graph_, 0), exec);

  exec.exec_ = std::shared_ptr<cudaGraphExec_t>(raw_exec, [](cudaGraphExec_t *const e) {
    if (e != nullptr && *e != nullptr) {
      cudaGraphExecDestroy(*e);
    }
    delete e;
  });
  return exec;
}

cudaGraph_t CudaGraph::graph() const {
  CUDA_CHECK(graph_ != nullptr);
  return *graph_;
}

// ==================== CudaGraphExec ====================

CudaGraphExec::CudaGraphExec() {
  auto raw_exec = new cudaGraphExec_t;
  *raw_exec = nullptr;
  exec_ = std::shared_ptr<cudaGraphExec_t>(raw_exec, [](cudaGraphExec_t *const e) { delete e; });
}

bool CudaGraphExec::Launch(cudaStream_t stream) const {
  CUDA_RETURN_VAL_IF(exec_ == nullptr, false);
  CUDART_RETURN_CALL(cudaGraphLaunch(*exec_, stream));
  return true;
}

bool CudaGraphExec::Launch(const stream::CudaStream &cuda_stream) const { return Launch(cuda_stream.stream()); }

bool CudaGraphExec::Update(const CudaGraph &graph) {
  CUDA_RETURN_VAL_IF(exec_ == nullptr, false);
  cudaGraphExecUpdateResultInfo resultInfo = {};
  CUDART_RETURN_CALL(cudaGraphExecUpdate(*exec_, graph.graph(), &resultInfo));
  if (resultInfo.result != cudaGraphExecUpdateSuccess) {
    LOG(ERROR) << "CUDA Graph Exec Update failed, result: " << resultInfo.result;
    return false;
  }
  return true;
}

cudaGraphExec_t CudaGraphExec::exec() const {
  CUDA_CHECK(exec_ != nullptr);
  return *exec_;
}

bool CudaGraphExec::operator==(const CudaGraphExec &other) const { return this->exec_ == other.exec_; }

}  // namespace graph
}  // namespace driver
