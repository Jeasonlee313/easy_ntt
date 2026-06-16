//
// Created by jeason on 1/28/26.
//

#include "driver/cuda_base_memory.h"

#include "driver/cuda_macros.h"

namespace {
static const std::shared_ptr<driver::stream::CudaStream> &DefaultStream() {
  static auto s =
      std::make_shared<driver::stream::CudaStream>(driver::stream::CudaStream::Flag::NON_BLOCKING);
  return s;
}
}  // namespace

namespace driver {
namespace base_memory {

CudaBaseMemory::CudaBaseMemory(const CudaMemoryType &memory_type) : type_(memory_type) {}

CudaBaseMemory::CudaBaseMemory(void *ptr, const CudaMemoryType &memory_type, size_t byte_size)
    : byte_size_(byte_size), capacity_bytes_(byte_size), is_view_(true), type_(memory_type) {
  if (type_ == CudaMemoryType::DEVICE) {
    device_ = ptr;
  } else if (type_ == CudaMemoryType::PINNED || type_ == CudaMemoryType::PINNED_WC) {
    host_ = ptr;
  }
}

CudaBaseMemory::~CudaBaseMemory() {
  if (!is_view_) {
    FreeMemory();
  }
}

bool CudaBaseMemory::Memset(const int32_t value, const size_t byte_offset, const size_t byte_size) {
  return Memset(value, byte_offset, byte_size, stream());
}

bool CudaBaseMemory::Memset(const int32_t value, const size_t byte_offset, const size_t byte_size,
                            const CudaStream &stream) {
  CUDA_RETURN_VAL_IF_QUIET(byte_size == 0, true);
  CUDA_RETURN_VAL_IF(byte_offset + byte_size > byte_size_, false);
  switch (type_) {
    case CudaMemoryType::PINNED:
    case CudaMemoryType::PINNED_WC:
      CUDA_CHECK_NOTNULL(host_);
      memset(reinterpret_cast<char *>(host_) + byte_offset, value, byte_size);
      break;
    case CudaMemoryType::DEVICE:
      CUDA_CHECK_NOTNULL(device_);
      CUDART_RETURN_VAL_IF_FAIL(
          cudaMemsetAsync(reinterpret_cast<char *>(device_) + byte_offset, value, byte_size, stream.stream()), false);
      break;
    default:
      return false;
  }
  return true;
}

bool CudaBaseMemory::MemcpyFromHost(const void *const data, const size_t byte_offset, const size_t byte_size) {
  return MemcpyFromHost(data, byte_offset, byte_size, stream());
}

bool CudaBaseMemory::MemcpyFromHost(const void *const data, const size_t byte_offset, const size_t byte_size,
                                    const CudaStream &stream) {
  CUDA_RETURN_VAL_IF_QUIET(byte_size == 0, true);
  CUDA_RETURN_VAL_IF(byte_offset + byte_size > byte_size_, false);
  switch (type_) {
    case CudaMemoryType::PINNED:
    case CudaMemoryType::PINNED_WC:
      CUDA_CHECK_NOTNULL(host_);
      if (data != host_) {
        VLOG(1) << "Additional CPU copies, considering simplifying the data flow";
        memcpy(reinterpret_cast<char *>(host_) + byte_offset, data, byte_size);
      }
      break;
    case CudaMemoryType::DEVICE:
      CUDA_CHECK_NOTNULL(device_);
      CUDART_RETURN_VAL_IF_FAIL(cudaMemcpyAsync(reinterpret_cast<char *>(device_) + byte_offset, data, byte_size,
                                                cudaMemcpyHostToDevice, stream.stream()),
                                false);
      break;
    default:
      return false;
  }
  return true;
}

bool CudaBaseMemory::MemcpyToHost(const size_t byte_offset, const size_t byte_size, void *data) const {
  CUDA_RETURN_VAL_IF(!MemcpyToHost(byte_offset, byte_size, stream(), data), false);
  // For device memory, ensure the async D2H copy is completed before returning
  if (type_ == CudaMemoryType::DEVICE) {
    CUDA_RETURN_VAL_IF(!stream().Synchronize(), false);
  }
  return true;
}

bool CudaBaseMemory::MemcpyToHost(const size_t byte_offset, const size_t byte_size, const CudaStream &stream,
                                  void *data) const {
  CUDA_RETURN_VAL_IF_QUIET(byte_size == 0, true);
  CUDA_RETURN_VAL_IF(byte_offset + byte_size > byte_size_, false);
  switch (type_) {
    case CudaMemoryType::PINNED:
    case CudaMemoryType::PINNED_WC:
      CUDA_CHECK_NOTNULL(host_);
      if (data != host_) {
        VLOG(1) << "Additional CPU copies, considering simplifying the data flow";
        memcpy(data, reinterpret_cast<char *>(host_) + byte_offset, byte_size);
      }
      break;
    case CudaMemoryType::DEVICE:
      CUDA_CHECK_NOTNULL(device_);
      CUDART_RETURN_VAL_IF_FAIL(cudaMemcpyAsync(data, reinterpret_cast<char *>(device_) + byte_offset, byte_size,
                                                cudaMemcpyDeviceToHost, stream.stream()),
                                false);
      break;
    default:
      return false;
  }
  return true;
}

bool CudaBaseMemory::MemcpyFromDevice(const void *const data, const size_t byte_offset, const size_t byte_size) {
  return MemcpyFromDevice(data, byte_offset, byte_size, stream());
}

bool CudaBaseMemory::MemcpyFromDevice(const void *const data, const size_t byte_offset, const size_t byte_size,
                                      const CudaStream &stream) {
  CUDA_RETURN_VAL_IF_QUIET(byte_size == 0, true);
  CUDA_RETURN_VAL_IF(byte_offset + byte_size > byte_size_, false);
  switch (type_) {
    case CudaMemoryType::PINNED:
    case CudaMemoryType::PINNED_WC:
      CUDA_CHECK_NOTNULL(host_);
      CUDART_RETURN_VAL_IF_FAIL(cudaMemcpyAsync(reinterpret_cast<char *>(host_) + byte_offset, data, byte_size,
                                                cudaMemcpyDeviceToHost, stream.stream()),
                                false);
      break;
    case CudaMemoryType::DEVICE:
      CUDA_CHECK_NOTNULL(device_);
      CUDART_RETURN_VAL_IF_FAIL(cudaMemcpyAsync(reinterpret_cast<char *>(device_) + byte_offset, data, byte_size,
                                                cudaMemcpyDeviceToDevice, stream.stream()),
                                false);
      break;
    default:
      return false;
  }
  return true;
}

bool CudaBaseMemory::MemcpyToDevice(const size_t byte_offset, const size_t byte_size, void *data) const {
  return MemcpyToDevice(byte_offset, byte_size, stream(), data);
}

bool CudaBaseMemory::MemcpyToDevice(const size_t byte_offset, const size_t byte_size, const CudaStream &stream,
                                    void *data) const {
  CUDA_RETURN_VAL_IF_QUIET(byte_size == 0, true);
  CUDA_RETURN_VAL_IF(byte_offset + byte_size > byte_size_, false);
  switch (type_) {
    case CudaMemoryType::PINNED:
    case CudaMemoryType::PINNED_WC:
      CUDA_CHECK_NOTNULL(host_);
      CUDART_RETURN_VAL_IF_FAIL(cudaMemcpyAsync(data, reinterpret_cast<char *>(host_) + byte_offset, byte_size,
                                                cudaMemcpyHostToDevice, stream.stream()),
                                false);
      break;
    case CudaMemoryType::DEVICE:
      CUDA_CHECK_NOTNULL(device_);
      CUDART_RETURN_VAL_IF_FAIL(cudaMemcpyAsync(data, reinterpret_cast<char *>(device_) + byte_offset, byte_size,
                                                cudaMemcpyDeviceToDevice, stream.stream()),
                                false);
      break;
    default:
      return false;
  }
  return true;
}

void CudaBaseMemory::FreeMemory() {
  switch (type_) {
    case CudaMemoryType::PINNED:
    case CudaMemoryType::PINNED_WC:
      if (host_registered_) {
        CUDART_CALL(cudaHostUnregister(host_));
        host_registered_ = false;
      } else if (host_ != nullptr) {
        CUDART_CALL(cudaFreeHost(host_));
      }
      break;
    case CudaMemoryType::DEVICE:
      if (device_ != nullptr) {
        if (use_async_allocator_) {
          CUDART_CALL(cudaFreeAsync(device_, stream().stream()));
        } else {
          CUDART_CALL(cudaFree(device_));
        }
      }
      break;
    default:
      break;
  }
  host_ = nullptr;
  device_ = nullptr;
  byte_size_ = 0;
  capacity_bytes_ = 0;
  host_registered_ = false;
  use_async_allocator_ = false;
}

bool CudaBaseMemory::AllocateMemory(const size_t byte_size) {
  size_t requested = byte_size;
  byte_size_ = requested;
  CUDA_RETURN_VAL_IF(byte_size_ == 0, true);
  // If existing capacity is sufficient, reuse without reallocation
  if (requested <= capacity_bytes_ && (device_ != nullptr || host_ != nullptr)) {
    return true;
  }
  // Otherwise free and reallocate
  FreeMemory();
  byte_size_ = requested;  // Restore after FreeMemory zeroed it
  switch (type_) {
    case CudaMemoryType::PINNED:
      CUDART_RETURN_VAL_IF_FAIL(cudaHostAlloc(&host_, byte_size_, 0), false);
      CUDA_CHECK_NOTNULL(host_);
      break;
    case CudaMemoryType::PINNED_WC:
      CUDART_RETURN_VAL_IF_FAIL(cudaHostAlloc(&host_, byte_size_, cudaHostAllocWriteCombined), false);
      CUDA_CHECK_NOTNULL(host_);
      break;
    case CudaMemoryType::DEVICE: {
      cudaError_t err = cudaMallocAsync(&device_, byte_size_, stream().stream());
      if (err != cudaSuccess) {
        LOG(WARNING) << "cudaMallocAsync failed: " << cudaGetErrorString(err) << ", fallback to cudaMalloc";
        cudaGetLastError();
        CUDART_RETURN_VAL_IF_FAIL(cudaMalloc(&device_, byte_size_), false);
        use_async_allocator_ = false;
      } else {
        use_async_allocator_ = true;
      }
      CUDA_CHECK_NOTNULL(device_);
      break;
    }
    default:
      return false;
  }
  capacity_bytes_ = byte_size_;
  return true;
}

bool CudaBaseMemory::AllocateMemoryAsync(const size_t byte_size) {
  CUDA_RETURN_VAL_IF(type_ != CudaMemoryType::DEVICE, false);
  return AllocateMemory(byte_size);
}

bool CudaBaseMemory::RegisterHostMemory(void *host_ptr, const size_t byte_size) {
  CUDA_RETURN_VAL_IF(host_ptr == nullptr, false);
  CUDA_RETURN_VAL_IF(type_ != CudaMemoryType::PINNED && type_ != CudaMemoryType::PINNED_WC, false);
  CUDART_RETURN_VAL_IF_FAIL(cudaHostRegister(host_ptr, byte_size, cudaHostRegisterDefault), false);
  host_ = host_ptr;
  byte_size_ = byte_size;
  host_registered_ = true;
  return true;
}

void CudaBaseMemory::SetStream(const std::shared_ptr<CudaStream> &stream) { stream_ = stream; }

void *CudaBaseMemory::ptr() const {
  switch (type_) {
    case CudaMemoryType::DEVICE:
      return device_;
    case CudaMemoryType::PINNED:
    case CudaMemoryType::PINNED_WC:
      return host_;
    default:
      CUERROR << "invalid memory type, return nullptr!";
      return nullptr;
  }
}

size_t CudaBaseMemory::byte_size() const { return byte_size_; }

size_t CudaBaseMemory::capacity_bytes() const { return capacity_bytes_; }

CudaMemoryType CudaBaseMemory::type() const { return type_; }

bool CudaBaseMemory::is_view() const { return is_view_; }

CudaStream CudaBaseMemory::stream() const {
  if (stream_ != nullptr) {
    return *stream_;
  }
  return *DefaultStream();
}

}  // namespace base_memory
}  // namespace driver
