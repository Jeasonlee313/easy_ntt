//
// Created by jeason on 5/9/26.
//

#include "driver/cuda_event.h"

#include "driver/cuda_macros.h"

namespace driver {
namespace event {

CudaEvent::CudaEvent() {
  auto raw_event = new cudaEvent_t;
  CUDART_RETURN_VAL_IF_FAIL(cudaEventCreate(raw_event), );
  event_ = std::shared_ptr<cudaEvent_t>(raw_event, [](cudaEvent_t *const e) {
    if (e != nullptr && *e != nullptr) {
      cudaEventDestroy(*e);
    }
    delete e;
  });
  flags_ = Flag::DEFAULT_TIMING;
}

CudaEvent::CudaEvent(const Flag &flags) : flags_(flags) {
  auto raw_event = new cudaEvent_t;
  switch (flags) {
    case Flag::DEFAULT:
      *raw_event = nullptr;
      event_ = std::shared_ptr<cudaEvent_t>(raw_event, [](cudaEvent_t *const e) { delete e; });
      break;
    default:
      CUDART_RETURN_VAL_IF_FAIL(cudaEventCreateWithFlags(raw_event, static_cast<unsigned int>(flags_)), );
      event_ = std::shared_ptr<cudaEvent_t>(raw_event, [](cudaEvent_t *const e) {
        if (e != nullptr && *e != nullptr) {
          cudaEventDestroy(*e);
        }
        delete e;
      });
  }
}

bool CudaEvent::Record(cudaStream_t stream) const {
  CUDA_RETURN_VAL_IF(event_ == nullptr, false);
  CUDART_RETURN_CALL(cudaEventRecord(*event_, stream));
  return true;
}

bool CudaEvent::Record(const stream::CudaStream &cuda_stream) const { return Record(cuda_stream.stream()); }

bool CudaEvent::Synchronize() const {
  CUDA_RETURN_VAL_IF(event_ == nullptr, false);
  CUDART_RETURN_CALL(cudaEventSynchronize(*event_));
  return true;
}

bool CudaEvent::Query() const {
  CUDA_RETURN_VAL_IF(event_ == nullptr, false);
  cudaError_t err = cudaEventQuery(*event_);
  if (err == cudaErrorNotReady) {
    return false;
  }
  CUDART_RETURN_IF_ERROR(err);
  return true;
}

float CudaEvent::ElapsedTime(const CudaEvent &start, const CudaEvent &end) {
  CUDA_RETURN_VAL_IF(start.event_ == nullptr || end.event_ == nullptr, -1.0f);
  float ms = -1.0f;
  CUDART_RETURN_VAL_IF_FAIL(cudaEventElapsedTime(&ms, *start.event_, *end.event_), -1.0f);
  return ms;
}

CudaEvent::Flag CudaEvent::flags() const { return flags_; }

cudaEvent_t CudaEvent::event() const {
  CUDA_CHECK(event_ != nullptr);
  return *event_;
}

bool CudaEvent::operator==(const CudaEvent &other) const { return this->event_ == other.event_; }

}  // namespace event
}  // namespace driver
