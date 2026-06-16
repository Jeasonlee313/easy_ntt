//
// Created by jeason on 1/28/26.
//

#ifndef TENSORRT_ENGINE_CUDA_STREAM_H
#define TENSORRT_ENGINE_CUDA_STREAM_H

#include <cuda_runtime_api.h>

#include <memory>

namespace driver {
namespace event {
class CudaEvent;
}
namespace stream {
class CudaStream {
 public:
  enum class Flag {
    DEFAULT = -1,
    BLOCKING = cudaStreamDefault,
    NON_BLOCKING = cudaStreamNonBlocking,
    HIGH_PRIORITY = 1,
    LOW_PRIORITY = 2
  };

  CudaStream();
  explicit CudaStream(const Flag &flags);
  explicit CudaStream(const int32_t priority);
  ~CudaStream() = default;

  [[nodiscard]] bool Synchronize() const;
  [[nodiscard]] bool WaitEvent(const cudaEvent_t event) const;
  [[nodiscard]] bool WaitEvent(const event::CudaEvent &cuda_event) const;

  [[nodiscard]] Flag flags() const;
  [[nodiscard]] cudaStream_t stream() const;

  bool operator==(const CudaStream &stream) const;

 protected:
  Flag flags_ = Flag::DEFAULT;
  std::shared_ptr<cudaStream_t> stream_ = nullptr;
};
}  // namespace stream
}  // namespace driver

#endif  // TENSORRT_ENGINE_CUDA_STREAM_H
