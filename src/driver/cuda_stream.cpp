//
// Created by jeason on 1/28/26.
//

#include "driver/cuda_stream.h"

#include "driver/cuda_event.h"
#include "driver/cuda_macros.h"

namespace driver {
namespace stream {

CudaStream::CudaStream() {
  auto raw_stream = new cudaStream_t;
  CUDART_RETURN_VAL_IF_FAIL(cudaStreamCreateWithFlags(raw_stream, cudaStreamNonBlocking), );
  stream_ = std::shared_ptr<cudaStream_t>(raw_stream, [](cudaStream_t *const stream) {
    cudaStreamDestroy(*stream);
    delete stream;
  });
  flags_ = Flag::NON_BLOCKING;
}

CudaStream::CudaStream(const Flag &flags) : flags_(flags) {
  auto raw_stream = new cudaStream_t;
  switch (flags) {
    case Flag::DEFAULT:
      *raw_stream = nullptr;
      stream_ = std::shared_ptr<cudaStream_t>(raw_stream, [](cudaStream_t *const stream) { delete stream; });
      break;
    case Flag::HIGH_PRIORITY:
    case Flag::LOW_PRIORITY: {
      static int least_priority = 0;
      static int greatest_priority = 0;
      static bool priority_cached = false;
      if (!priority_cached) {
        CUDART_RETURN_VAL_IF_FAIL(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority), );
        priority_cached = true;
      }
      int priority = (flags == Flag::HIGH_PRIORITY) ? greatest_priority : least_priority;
      CUDART_RETURN_VAL_IF_FAIL(cudaStreamCreateWithPriority(raw_stream, cudaStreamNonBlocking, priority), );
      stream_ = std::shared_ptr<cudaStream_t>(raw_stream, [](cudaStream_t *const stream) {
        cudaStreamDestroy(*stream);
        delete stream;
      });
      break;
    }
    default:
      CUDART_RETURN_VAL_IF_FAIL(cudaStreamCreateWithFlags(raw_stream, static_cast<int32_t>(flags_)), );
      stream_ = std::shared_ptr<cudaStream_t>(raw_stream, [](cudaStream_t *const stream) {
        cudaStreamDestroy(*stream);
        delete stream;
      });
  }
}

CudaStream::CudaStream(const int32_t priority) {
  auto raw_stream = new cudaStream_t;
  CUDART_RETURN_VAL_IF_FAIL(cudaStreamCreateWithPriority(raw_stream, cudaStreamNonBlocking, priority), );
  stream_ = std::shared_ptr<cudaStream_t>(raw_stream, [](cudaStream_t *const stream) {
    cudaStreamDestroy(*stream);
    delete stream;
  });
  flags_ = Flag::NON_BLOCKING;
}

bool CudaStream::Synchronize() const {
  CUDA_RETURN_VAL_IF(stream_ == nullptr, false);
  CUDART_RETURN_CALL(cudaStreamSynchronize(*stream_));
  return true;
}

bool CudaStream::WaitEvent(const cudaEvent_t event) const {
  CUDA_RETURN_VAL_IF(stream_ == nullptr, false);
  CUDART_RETURN_CALL(cudaStreamWaitEvent(*stream_, event, 0));
  return true;
}

bool CudaStream::WaitEvent(const event::CudaEvent &cuda_event) const { return WaitEvent(cuda_event.event()); }

CudaStream::Flag CudaStream::flags() const { return flags_; }

cudaStream_t CudaStream::stream() const {
  CUDA_CHECK(stream_ != nullptr);
  return *stream_;
}

bool CudaStream::operator==(const CudaStream &stream) const { return this->stream_ == stream.stream_; }

}  // namespace stream
}  // namespace driver
