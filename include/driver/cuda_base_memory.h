//
// Created by jeason on 1/28/26.
//

#ifndef TENSORRT_ENGINE_CUDA_BASE_MEMORY_H
#define TENSORRT_ENGINE_CUDA_BASE_MEMORY_H

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "driver/cuda_memory_type.h"
#include "driver/cuda_stream.h"

namespace driver {
namespace base_memory {
using driver::stream::CudaStream;

class CudaBaseMemory {
 public:
  explicit CudaBaseMemory(const CudaMemoryType &memory_type);
  //! @brief Create a memory view that does not own the underlying pointer
  CudaBaseMemory(void *ptr, const CudaMemoryType &memory_type, size_t byte_size);
  ~CudaBaseMemory();
  CudaBaseMemory(const CudaBaseMemory &) = delete;
  CudaBaseMemory &operator=(const CudaBaseMemory &memory) = delete;
  CudaBaseMemory(CudaBaseMemory &&) = delete;
  CudaBaseMemory &operator=(CudaBaseMemory &&memory) = delete;

  bool Memset(const int32_t value, const size_t byte_offset, const size_t byte_size);
  bool Memset(const int32_t value, const size_t byte_offset, const size_t byte_size, const CudaStream &stream);

  bool MemcpyFromHost(const void *const data, const size_t byte_offset, const size_t byte_size);
  bool MemcpyFromHost(const void *const data, const size_t byte_offset, const size_t byte_size,
                      const CudaStream &stream);

  bool MemcpyToHost(const size_t byte_offset, const size_t byte_size, void *data) const;
  bool MemcpyToHost(const size_t byte_offset, const size_t byte_size, const CudaStream &stream, void *data) const;

  bool MemcpyFromDevice(const void *const data, const size_t byte_offset, const size_t byte_size);
  bool MemcpyFromDevice(const void *const data, const size_t byte_offset, const size_t byte_size,
                        const CudaStream &stream);

  bool MemcpyToDevice(const size_t byte_offset, const size_t byte_size, void *data) const;
  bool MemcpyToDevice(const size_t byte_offset, const size_t byte_size, const CudaStream &stream, void *data) const;

  void FreeMemory();
  bool AllocateMemory(const size_t byte_size);
  bool AllocateMemoryAsync(const size_t byte_size);
  bool RegisterHostMemory(void *host_ptr, const size_t byte_size);

  void SetStream(const std::shared_ptr<CudaStream> &stream);

  [[nodiscard]] void *ptr() const;

  [[nodiscard]] size_t byte_size() const;
  [[nodiscard]] size_t capacity_bytes() const;
  [[nodiscard]] CudaMemoryType type() const;
  [[nodiscard]] bool is_view() const;
  [[nodiscard]] CudaStream stream() const;

 protected:
  void *device_ = nullptr;
  void *host_ = nullptr;
  size_t byte_size_ = 0;
  size_t capacity_bytes_ = 0;
  const int32_t device_id_ = 0;
  bool host_registered_ = false;
  bool use_async_allocator_ = false;
  bool is_view_ = false;
  CudaMemoryType type_ = CudaMemoryType::INVALID;
  std::shared_ptr<CudaStream> stream_ = nullptr;
};
}  // namespace base_memory
}  // namespace driver

#endif  // TENSORRT_ENGINE_CUDA_BASE_MEMORY_H
