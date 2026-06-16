//
// Created by jeason on 2/5/26.
//

#include "tensorrt_engine/tensor.h"

namespace tensorrt_engine {
namespace tensor {

Tensor::Tensor() : Tensor(nullptr) {}

Tensor::Tensor(const std::shared_ptr<CudaStream> &stream) {
  data_ = std::make_shared<CudaBaseMemory>(CudaMemoryType::DEVICE);
  if (stream) {
    SetStream(stream);
  }
}

Tensor::Tensor(const nvinfer1::Dims &dims, nvinfer1::DataType data_type) : Tensor(dims, data_type, nullptr) {}

Tensor::Tensor(const nvinfer1::Dims &dims, nvinfer1::DataType data_type, const std::shared_ptr<CudaStream> &stream)
    : Tensor(stream) {
  Resize(dims, data_type);
}

Tensor::Tensor(const nvinfer1::Dims &dims, nvinfer1::DataType data_type, CudaMemoryType mem_type)
    : Tensor(dims, data_type, mem_type, nullptr) {}

Tensor::Tensor(const nvinfer1::Dims &dims, nvinfer1::DataType data_type, CudaMemoryType mem_type,
               const std::shared_ptr<CudaStream> &stream) {
  data_ = std::make_shared<CudaBaseMemory>(mem_type);
  if (stream) {
    SetStream(stream);
  }
  Resize(dims, data_type);
}

bool Tensor::CopyFromHost(const void *data) { return data_->MemcpyFromHost(data, 0, byte_size_); }

bool Tensor::CopyToHost(void *data) const { return data_->MemcpyToHost(0, byte_size_, data); }

bool Tensor::CopyFromDevice(const void *data) { return data_->MemcpyFromDevice(data, 0, byte_size_); }

bool Tensor::CopyToDevice(void *data) const { return data_->MemcpyToDevice(0, byte_size_, data); }

bool Tensor::CopyFromHostAsync(const void *data, const CudaStream &stream) {
  return data_->MemcpyFromHost(data, 0, byte_size_, stream);
}

bool Tensor::CopyToHostAsync(void *data, const CudaStream &stream) const {
  return data_->MemcpyToHost(0, byte_size_, stream, data);
}

bool Tensor::CopyFromDeviceAsync(const void *data, const CudaStream &stream) {
  return data_->MemcpyFromDevice(data, 0, byte_size_, stream);
}

bool Tensor::CopyToDeviceAsync(void *data, const CudaStream &stream) const {
  return data_->MemcpyToDevice(0, byte_size_, stream, data);
}

const void *Tensor::data() const { return data_->ptr(); }

void *Tensor::mutable_data() { return data_->ptr(); }

void Tensor::SetStream(const std::shared_ptr<CudaStream> &stream) {
  stream_ = stream;
  data_->SetStream(stream);
}

CudaStream Tensor::stream() const {
  if (stream_) {
    return *stream_;
  }
  return data_->stream();
}

void Tensor::SetName(const std::string &name) { name_ = name; }

const std::string &Tensor::name() const { return name_; }

void Tensor::SetDataType(const nvinfer1::DataType &data_type) {
  data_type_ = data_type;
  switch (data_type_) {
    case nvinfer1::DataType::kFLOAT:
      element_size_ = 4;
      break;
    case nvinfer1::DataType::kHALF:
      element_size_ = 2;
      break;
    case nvinfer1::DataType::kBF16:
      element_size_ = 2;
      break;
    case nvinfer1::DataType::kINT8:
      element_size_ = 1;
      break;
    case nvinfer1::DataType::kINT32:
      element_size_ = 4;
      break;
    case nvinfer1::DataType::kINT64:
      element_size_ = 8;
      break;
    case nvinfer1::DataType::kBOOL:
      element_size_ = 1;
      break;
    default:
      element_size_ = 0;
      LOG(ERROR) << "Tensor::SetDataType: unsupported data type " << static_cast<int>(data_type_);
      break;
  }
}

const nvinfer1::DataType &Tensor::dataType() const { return data_type_; }

const nvinfer1::Dims &Tensor::dims() const { return dims_; }

bool Tensor::Resize(const nvinfer1::Dims &dims, nvinfer1::DataType data_type) {
  dims_ = dims;
  SetDataType(data_type);
  if (element_size_ == 0) {
    LOG(ERROR) << "Tensor::Resize: unsupported data type";
    return false;
  }
  num_elements_ = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      LOG(ERROR) << "Tensor::Resize: dynamic dimension (" << dims.d[i] << ") not supported in this overload";
      return false;
    }
    num_elements_ *= static_cast<size_t>(dims.d[i]);
  }
  byte_size_ = num_elements_ * element_size_;
  bool ok = data_->AllocateMemory(byte_size_);
  capacity_bytes_ = data_->capacity_bytes();
  return ok;
}

bool Tensor::Resize(size_t byte_size) {
  CUDA_RETURN_VAL_IF(data_ == nullptr, false);
  byte_size_ = byte_size;
  if (element_size_ > 0) {
    num_elements_ = byte_size / element_size_;
  }
  bool ok = data_->AllocateMemory(byte_size_);
  capacity_bytes_ = data_->capacity_bytes();
  return ok;
}

bool Tensor::Reshape(const nvinfer1::Dims &dims) {
  if (element_size_ == 0) {
    LOG(ERROR) << "Tensor::Reshape: element_size is 0, data type not set";
    return false;
  }
  size_t new_num_elements = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      LOG(ERROR) << "Tensor::Reshape: dynamic dimension not supported";
      return false;
    }
    new_num_elements *= static_cast<size_t>(dims.d[i]);
  }
  size_t new_byte_size = new_num_elements * element_size_;
  if (new_byte_size > capacity_bytes_) {
    LOG(ERROR) << "Tensor::Reshape: new byte size (" << new_byte_size << ") exceeds capacity (" << capacity_bytes_
               << ")";
    return false;
  }
  dims_ = dims;
  num_elements_ = new_num_elements;
  byte_size_ = new_byte_size;
  return true;
}

Tensor Tensor::View(const nvinfer1::Dims &dims) const {
  Tensor view;
  view.data_ = data_;  // share underlying CudaBaseMemory
  view.stream_ = stream_;
  view.name_ = name_ + "_view";
  view.data_type_ = data_type_;
  view.element_size_ = element_size_;
  view.capacity_bytes_ = capacity_bytes_;
  // Compute new byte_size from dims
  size_t new_num_elements = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    new_num_elements *= static_cast<size_t>(dims.d[i]);
  }
  view.dims_ = dims;
  view.num_elements_ = new_num_elements;
  view.byte_size_ = new_num_elements * element_size_;
  return view;
}

Tensor Tensor::ViewFrom(void *ptr, const nvinfer1::Dims &dims, nvinfer1::DataType data_type, CudaMemoryType mem_type,
                        const std::shared_ptr<CudaStream> &stream) {
  Tensor t;
  size_t num_elements = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    num_elements *= static_cast<size_t>(dims.d[i]);
  }
  size_t elem_size = 0;
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      elem_size = 4;
      break;
    case nvinfer1::DataType::kHALF:
      elem_size = 2;
      break;
    case nvinfer1::DataType::kBF16:
      elem_size = 2;
      break;
    case nvinfer1::DataType::kINT8:
      elem_size = 1;
      break;
    case nvinfer1::DataType::kINT32:
      elem_size = 4;
      break;
    case nvinfer1::DataType::kINT64:
      elem_size = 8;
      break;
    case nvinfer1::DataType::kBOOL:
      elem_size = 1;
      break;
    default:
      elem_size = 0;
      break;
  }
  size_t total_bytes = num_elements * elem_size;
  t.data_ = std::make_shared<CudaBaseMemory>(ptr, mem_type, total_bytes);
  if (stream) {
    t.stream_ = stream;
    t.data_->SetStream(stream);
  }
  t.dims_ = dims;
  t.data_type_ = data_type;
  t.element_size_ = elem_size;
  t.num_elements_ = num_elements;
  t.byte_size_ = total_bytes;
  t.capacity_bytes_ = total_bytes;
  return t;
}

void Tensor::SetIoMode(const nvinfer1::TensorIOMode &io_mode) { io_mode_ = io_mode; }

bool Tensor::is_input() const { return io_mode_ == nvinfer1::TensorIOMode::kINPUT; }

bool Tensor::is_output() const { return io_mode_ == nvinfer1::TensorIOMode::kOUTPUT; }

size_t Tensor::element_size() const { return element_size_; }

size_t Tensor::num_elements() const { return num_elements_; }

size_t Tensor::byte_size() const { return byte_size_; }

size_t Tensor::capacity_bytes() const { return capacity_bytes_; }

}  // namespace tensor
}  // namespace tensorrt_engine
