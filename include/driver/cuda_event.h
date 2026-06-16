//
// Created by jeason on 5/9/26.
//

#ifndef TENSORRT_ENGINE_CUDA_EVENT_H
#define TENSORRT_ENGINE_CUDA_EVENT_H

#include <cuda_runtime_api.h>

#include <memory>

#include "driver/cuda_stream.h"

namespace driver {
namespace event {

class CudaEvent {
 public:
  enum class Flag {
    DEFAULT = -1,
    DEFAULT_TIMING = 0,
    BLOCKING_SYNC = cudaEventBlockingSync,
    DISABLE_TIMING = cudaEventDisableTiming,
    INTERPROCESS = cudaEventInterprocess
  };

  CudaEvent();
  explicit CudaEvent(const Flag &flags);
  ~CudaEvent() = default;

  // Record event on the specified stream
  [[nodiscard]] bool Record(cudaStream_t stream) const;
  [[nodiscard]] bool Record(const stream::CudaStream &cuda_stream) const;

  // Synchronize on this event
  [[nodiscard]] bool Synchronize() const;

  // Non-blocking query: true if completed, false if not ready or error
  [[nodiscard]] bool Query() const;

  // Elapsed time between two events in milliseconds, returns -1.0f on failure
  [[nodiscard]] static float ElapsedTime(const CudaEvent &start, const CudaEvent &end);

  [[nodiscard]] Flag flags() const;
  [[nodiscard]] cudaEvent_t event() const;

  bool operator==(const CudaEvent &other) const;

 protected:
  Flag flags_ = Flag::DEFAULT;
  std::shared_ptr<cudaEvent_t> event_ = nullptr;
};

}  // namespace event
}  // namespace driver

#endif  // TENSORRT_ENGINE_CUDA_EVENT_H
