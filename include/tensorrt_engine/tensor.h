//
// Created by jeason on 2/5/26.
//

#ifndef TENSORRT_ENGINE_TENSOR_H
#define TENSORRT_ENGINE_TENSOR_H
#include <NvInfer.h>

#include "driver/cuda_base_memory.h"
#include "driver/cuda_macros.h"
#include "driver/cuda_stream.h"

namespace tensorrt_engine {
namespace tensor {
using driver::base_memory::CudaBaseMemory;
using driver::stream::CudaStream;

class Tensor {
 public:
  Tensor();
  explicit Tensor(const std::shared_ptr<CudaStream> &stream);

  Tensor(const nvinfer1::Dims &dims, nvinfer1::DataType data_type);
  Tensor(const nvinfer1::Dims &dims, nvinfer1::DataType data_type, const std::shared_ptr<CudaStream> &stream);
  Tensor(const nvinfer1::Dims &dims, nvinfer1::DataType data_type, CudaMemoryType mem_type);
  Tensor(const nvinfer1::Dims &dims, nvinfer1::DataType data_type, CudaMemoryType mem_type,
         const std::shared_ptr<CudaStream> &stream);
  ~Tensor() = default;

  // 数据访问 —— 统一返回 void*
  const void *data() const;
  void *mutable_data();

  // 拷贝
  bool CopyFromHost(const void *data);
  bool CopyToHost(void *data) const;
  bool CopyFromDevice(const void *data);
  bool CopyToDevice(void *data) const;
  bool CopyFromHostAsync(const void *data, const CudaStream &stream);
  bool CopyToHostAsync(void *data, const CudaStream &stream) const;
  bool CopyFromDeviceAsync(const void *data, const CudaStream &stream);
  bool CopyToDeviceAsync(void *data, const CudaStream &stream) const;

  // Resize / Reshape / View
  bool Resize(const nvinfer1::Dims &dims, nvinfer1::DataType data_type);  // 自动算 element_size，可能重新分配
  bool Resize(size_t byte_size);                                          // 直接按字节，可能重新分配
  //! @brief Reshape in-place without reallocation. Returns false if new size exceeds capacity.
  bool Reshape(const nvinfer1::Dims &dims);
  //! @brief Create a view Tensor sharing the same underlying memory.
  Tensor View(const nvinfer1::Dims &dims) const;
  //! @brief Create a Tensor from an external pointer (non-owning).
  static Tensor ViewFrom(void *ptr, const nvinfer1::Dims &dims, nvinfer1::DataType data_type, CudaMemoryType mem_type,
                         const std::shared_ptr<CudaStream> &stream = nullptr);

  // Stream
  void SetStream(const std::shared_ptr<CudaStream> &stream);
  CudaStream stream() const;

  // 元信息
  void SetName(const std::string &name);
  const std::string &name() const;
  void SetDataType(const nvinfer1::DataType &data_type);
  const nvinfer1::DataType &dataType() const;
  const nvinfer1::Dims &dims() const;
  void SetIoMode(const nvinfer1::TensorIOMode &io_mode);
  bool is_input() const;
  bool is_output() const;

  // 工具方法
  size_t element_size() const;
  size_t num_elements() const;
  size_t byte_size() const;
  size_t capacity_bytes() const;

  // LLM-oriented extensions (Module 7)
  //! @brief Reshape in-place without reallocation (alias for Reshape).
  bool reshape(const nvinfer1::Dims &dims) { return Reshape(dims); }

  //! @brief Get shape as TRT Dims (alias for dims).
  nvinfer1::Dims getShape() const { return dims(); }

  //! @brief Get memory capacity in bytes (alias for capacity_bytes).
  int64_t getMemoryCapacity() const { return static_cast<int64_t>(capacity_bytes()); }

  //! @brief Get raw data pointer (alias for mutable_data).
  void *rawPointer() { return mutable_data(); }
  void const *rawPointer() const { return data(); }

  //! @brief Get typed data pointer.
  template <typename T>
  T *dataPointer() {
    return reinterpret_cast<T *>(mutable_data());
  }
  template <typename T>
  T const *dataPointer() const {
    return reinterpret_cast<T const *>(data());
  }

 private:
  std::shared_ptr<CudaBaseMemory> data_;
  std::shared_ptr<CudaStream> stream_;
  std::string name_;
  nvinfer1::Dims dims_{};
  nvinfer1::DataType data_type_;
  nvinfer1::TensorIOMode io_mode_;
  size_t num_elements_ = 0;
  size_t element_size_ = 0;
  size_t byte_size_ = 0;
  size_t capacity_bytes_ = 0;
};

}  // namespace tensor
}  // namespace tensorrt_engine

#endif  // TENSORRT_ENGINE_TENSOR_H
