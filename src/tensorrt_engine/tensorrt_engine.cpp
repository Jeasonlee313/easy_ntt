//
// Created by jeason on 2/5/26.
//

#include <filesystem>

#include "tensorrt_engine/tensorrt_engine.hpp"
#include "tensorrt_engine/tensorrt_utility.hpp"
#include "tensorrt_engine/tensor.h"
#include "tensorrt_engine/trt_logger.h"
#include "driver/cuda_stream.h"
#include "driver/cuda_memory_type.h"
#include "nvtx/nvtx_scoped.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace tensorrt_engine {

using driver::stream::CudaStream;
using tensorrt_engine::logger::Logger;
using tensorrt_engine::tensor::Tensor;

// ── Constructor ──────────────────────────────────────────────────────

TensorRTEngine::TensorRTEngine(const std::string &engine_path, const TensorRTEngineConfig &config)
    : TensorRTEngine(engine_path, "", config) {}

TensorRTEngine::TensorRTEngine(const std::string &engine_path, const std::string &plugin_path,
                               const TensorRTEngineConfig &config)
    : config_(config) {
  plugin_handle_ = nullptr;
  if (!config_.enable_persistent_io_buffers) {
    config_.allocate_max_profile_buffers = false;
    config_.auto_freeze_io = false;
    config_.enable_cuda_graph = false;
  }
  cuda_graph_enabled_ = config_.enable_cuda_graph;

  if (!plugin_path.empty()) {
    LoadPluginLibrary(plugin_path);
  }
  try {
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file.is_open()) {
      LOG(ERROR) << "Failed to open engine file: " << engine_path;
      return;
    }

    engine_name_ = std::filesystem::path(engine_path).filename().string();

    engine_file.seekg(0, std::ios::end);
    size_t engine_size = engine_file.tellg();
    CUINFO << "Engine size: " << engine_size;
    engine_file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(engine_size);
    engine_file.read(engine_data.data(), engine_size);
    engine_file.close();

    size_t poolSize = 1024 * 1024 * 1024;  // 1 GB
    CUDART_CALL(cudaDeviceSetLimit(cudaLimitMallocHeapSize, poolSize));

    runtime_ = std::shared_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(logger_), [](nvinfer1::IRuntime* r) { delete r; });
    runtime_->setGpuAllocator(nullptr);

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime_->deserializeCudaEngine(engine_data.data(), engine_size),
        [](nvinfer1::ICudaEngine *engine) { delete engine; });

    if (!engine_) {
      CUERROR << "Failed to deserialize engine";
      return;
    }

    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
        engine_->createExecutionContext(),
        [](nvinfer1::IExecutionContext *context) { delete context; });

    if (!context_) {
      CUERROR << "Failed to create execution context";
      return;
    }

    context_->setOptimizationProfileAsync(0, stream_->stream());

    int32_t nb_aux = engine_->getNbAuxStreams();
    if (nb_aux > 0) {
      aux_streams_.resize(nb_aux, nullptr);
      for (int32_t i = 0; i < nb_aux; ++i) {
        cudaStreamCreate(&aux_streams_[i]);
      }
      context_->setAuxStreams(aux_streams_.data(), nb_aux);
      CUINFO << "Configured " << nb_aux << " auxiliary stream(s) for multi-stream engine";
    }

    // Detect whether this engine has any dynamic bindings.
    engine_has_dynamic_bindings_ = false;
    for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
      const std::string name = engine_->getIOTensorName(i);
      if (IsDynamicBinding(name)) {
        engine_has_dynamic_bindings_ = true;
        break;
      }
    }

    // Allocate I/O bindings in constructor — output shapes resolved via pre-setting inputs
    AllocateIOBindingsMemory();
    if (config_.auto_freeze_io && !FreezeIO()) {
      LOG(WARNING) << "TensorRTEngine auto_freeze_io requested but FreezeIO failed";
    }
  } catch (const std::exception &e) {
    LOG(ERROR) << "Failed to create engine: " << e.what();
  }
}

// ── Destructor ───────────────────────────────────────────────────────

TensorRTEngine::~TensorRTEngine() { Cleanup(); }

// ── AllocateIOBindingsMemory ─────────────────────────────────────────

void TensorRTEngine::AllocateIOBindingsMemory() {
  auto format_dims = [](const nvinfer1::Dims &dims) {
    std::stringstream ss;
    for (int32_t d = 0; d < dims.nbDims; ++d) {
      if (d < dims.nbDims - 1) ss << dims.d[d] << " * ";
      else ss << dims.d[d];
    }
    return ss.str();
  };

  auto ensure_concrete = [this, &format_dims](const std::string &name, const nvinfer1::Dims &shape,
                                              const char *phase) -> nvinfer1::Dims {
    nvinfer1::Dims concrete = shape;
    if (!HasConcreteDims(concrete)) {
      LOG(WARNING) << phase << " " << name << ": shape has unresolved dims (" << format_dims(concrete)
                   << "), replacing invalid dims with 1";
      for (int32_t j = 0; j < concrete.nbDims; ++j) {
        if (concrete.d[j] <= 0) concrete.d[j] = 1;
      }
    }
    return concrete;
  };

  // Phase 1: allocate inputs at max profile shape (dynamic) or engine shape (static),
  //          set context input shapes and bind addresses.
  for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
    const std::string name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name.c_str()) != nvinfer1::TensorIOMode::kINPUT) continue;

    const auto tensor_type = engine_->getTensorDataType(name.c_str());
    const auto tensor_location = engine_->getTensorLocation(name.c_str());
    const nvinfer1::Dims engine_shape = engine_->getTensorShape(name.c_str());
    const bool is_dynamic = IsDynamicBinding(name);

    nvinfer1::Dims alloc_shape = engine_shape;
    if (is_dynamic && config_.allocate_max_profile_buffers) {
      nvinfer1::Dims profile_shape = GetProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
      if (HasConcreteDims(profile_shape) && profile_shape.nbDims == engine_shape.nbDims) {
        alloc_shape = profile_shape;
      } else {
        LOG(WARNING) << "Input " << name << ": max profile shape invalid, falling back to engine shape";
      }
    }
    alloc_shape = ensure_concrete(name, alloc_shape, "Input");

    std::shared_ptr<Tensor> tensor_ptr;
    if (tensor_location == nvinfer1::TensorLocation::kHOST) {
      tensor_ptr = std::make_shared<Tensor>(alloc_shape, tensor_type, CudaMemoryType::PINNED, stream_);
    } else {
      tensor_ptr = std::make_shared<Tensor>(alloc_shape, tensor_type, stream_);
    }
    tensor_ptr->SetName(name);
    tensor_ptr->SetIoMode(nvinfer1::TensorIOMode::kINPUT);
    tensor_map_[name] = tensor_ptr;

    void *ptr = tensor_ptr->mutable_data();
    if (ptr != nullptr) {
      if (!context_->setTensorAddress(name.c_str(), ptr)) {
        LOG(WARNING) << "Failed to set tensor address for input: " << name;
      }
    } else {
      LOG(WARNING) << "Tensor memory not allocated for input binding: " << name;
    }

    if (is_dynamic) {
      if (!context_->setInputShape(name.c_str(), alloc_shape)) {
        LOG(WARNING) << "Failed to set input shape for: " << name;
      }
    }

    CUINFO << "Binding name: " << name << ", is input: true"
           << ", location: " << (tensor_location == nvinfer1::TensorLocation::kHOST ? "HOST" : "DEVICE")
           << ", with shape dims nbDims: " << engine_shape.nbDims << " (" << format_dims(engine_shape) << " )"
           << ", data type: " << utility::dataTypeToString(tensor_type)
           << (is_dynamic && config_.allocate_max_profile_buffers ? " [dynamic, allocated at max profile]" : "");
  }

  // Phase 2: infer output shapes from context (after all inputs are set) and allocate outputs.
  for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
    const std::string name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name.c_str()) != nvinfer1::TensorIOMode::kOUTPUT) continue;

    const auto tensor_type = engine_->getTensorDataType(name.c_str());
    const auto tensor_location = engine_->getTensorLocation(name.c_str());
    const nvinfer1::Dims engine_shape = engine_->getTensorShape(name.c_str());
    const bool is_dynamic = IsDynamicBinding(name);

    nvinfer1::Dims alloc_shape = context_->getTensorShape(name.c_str());
    if (!HasConcreteDims(alloc_shape)) {
      LOG(WARNING) << "Output " << name << ": shape inference returned unresolved shape (" << format_dims(alloc_shape)
                   << ")";
      if (config_.allocate_max_profile_buffers) {
        LOG(WARNING) << "Output " << name << ": falling back to max profile shape";
        alloc_shape = GetProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
      }
      if (!HasConcreteDims(alloc_shape)) {
        LOG(WARNING) << "Output " << name << ": using engine shape with fallback 1";
        alloc_shape = engine_shape;
      }
    }

    // If the inferred/profile shape has fewer dims than the engine shape, merge the static
    // dimensions from the engine shape so that e.g. a profile that only gives the dynamic
    // axis size (81) is expanded back to the full shape (81 * 4096).
    if (alloc_shape.nbDims > 0 && alloc_shape.nbDims < engine_shape.nbDims) {
      nvinfer1::Dims merged = engine_shape;
      for (int32_t j = 0; j < engine_shape.nbDims && j < alloc_shape.nbDims; ++j) {
        if (merged.d[j] < 0) {
          merged.d[j] = alloc_shape.d[j];
        }
      }
      alloc_shape = merged;
    }

    alloc_shape = ensure_concrete(name, alloc_shape, "Output");

    std::shared_ptr<Tensor> tensor_ptr;
    if (tensor_location == nvinfer1::TensorLocation::kHOST) {
      tensor_ptr = std::make_shared<Tensor>(alloc_shape, tensor_type, CudaMemoryType::PINNED, stream_);
    } else {
      tensor_ptr = std::make_shared<Tensor>(alloc_shape, tensor_type, stream_);
    }
    tensor_ptr->SetName(name);
    tensor_ptr->SetIoMode(nvinfer1::TensorIOMode::kOUTPUT);
    tensor_map_[name] = tensor_ptr;

    void *ptr = tensor_ptr->mutable_data();
    if (ptr != nullptr) {
      if (!context_->setTensorAddress(name.c_str(), ptr)) {
        LOG(WARNING) << "Failed to set tensor address for output: " << name;
      }
    } else {
      LOG(WARNING) << "Tensor memory not allocated for output binding: " << name;
    }

    CUINFO << "Binding name: " << name << ", is input: false"
           << ", location: " << (tensor_location == nvinfer1::TensorLocation::kHOST ? "HOST" : "DEVICE")
           << ", with shape dims nbDims: " << engine_shape.nbDims << " (" << format_dims(engine_shape) << " )"
           << ", data type: " << utility::dataTypeToString(tensor_type)
           << (is_dynamic && config_.allocate_max_profile_buffers ? " [dynamic, allocated at max profile]" : "");
  }
}

// ── Cleanup ──────────────────────────────────────────────────────────

void TensorRTEngine::Cleanup() {
  ResetCudaGraph();
  tensor_map_.clear();
  user_binding_addresses_.clear();
  input_ready_events_.clear();
  output_ready_events_.clear();
  context_.reset();
  engine_.reset();
  runtime_.reset();
  for (cudaStream_t aux : aux_streams_) {
    if (aux != nullptr) cudaStreamDestroy(aux);
  }
  aux_streams_.clear();
}

// ── IsLoaded ─────────────────────────────────────────────────────────

bool TensorRTEngine::IsLoaded() const { return (engine_ != nullptr && context_ != nullptr); }

// ── Stream management ────────────────────────────────────────────────

bool TensorRTEngine::SetStream(cudaStream_t stream) {
  if (!stream) return false;
  ResetCudaGraph();
  external_stream_ = stream;
  return true;
}

cudaStream_t TensorRTEngine::GetCudaStream() const {
  if (!engine_ || !context_) return nullptr;
  return external_stream_ ? external_stream_ : stream_->stream();
}

bool TensorRTEngine::Synchronize() const {
  if (!engine_ || !context_) return true;  // no-op on uninitialized engine
  cudaStream_t s = GetCudaStream();
  return (s != nullptr) ? (cudaStreamSynchronize(s) == cudaSuccess) : false;
}

cudaEvent_t TensorRTEngine::GetInputReadyEvent(const std::string &name) const {
  auto it = input_ready_events_.find(name);
  return it != input_ready_events_.end() ? it->second->event() : nullptr;
}

cudaEvent_t TensorRTEngine::GetOutputReadyEvent(const std::string &name) const {
  auto it = output_ready_events_.find(name);
  return it != output_ready_events_.end() ? it->second->event() : nullptr;
}

bool TensorRTEngine::WaitInputReady(const std::string &name, cudaStream_t stream) const {
  auto it = input_ready_events_.find(name);
  if (it == input_ready_events_.end()) {
    LOG(ERROR) << "Input ready event not found: " << name;
    return false;
  }
  return WaitEvent(stream, it->second);
}

bool TensorRTEngine::WaitOutputReady(const std::string &name, cudaStream_t stream) const {
  auto it = output_ready_events_.find(name);
  if (it == output_ready_events_.end()) {
    LOG(ERROR) << "Output ready event not found: " << name;
    return false;
  }
  return WaitEvent(stream, it->second);
}

bool TensorRTEngine::SynchronizeInput(const std::string &name) const {
  auto it = input_ready_events_.find(name);
  if (it == input_ready_events_.end()) {
    LOG(ERROR) << "Input ready event not found: " << name;
    return false;
  }
  return it->second->Synchronize();
}

bool TensorRTEngine::SynchronizeOutput(const std::string &name) const {
  auto it = output_ready_events_.find(name);
  if (it == output_ready_events_.end()) {
    LOG(ERROR) << "Output ready event not found: " << name;
    return false;
  }
  return it->second->Synchronize();
}

void TensorRTEngine::EnableCudaGraph(bool enable) {
  cuda_graph_enabled_ = enable;
  config_.enable_cuda_graph = enable;
  if (!cuda_graph_enabled_) {
    ResetCudaGraph();
  }
}

bool TensorRTEngine::CaptureCudaGraph() {
  if (!engine_ || !context_) {
    CUERROR << "Engine not loaded";
    return false;
  }
  cudaStream_t exec_stream = GetCudaStream();
  for (const auto &pair : tensor_map_) {
    if (!pair.second->is_input()) continue;
    auto event_it = input_ready_events_.find(pair.first);
    if (event_it != input_ready_events_.end() && !WaitEvent(exec_stream, event_it->second)) {
      LOG(ERROR) << "Failed to wait input ready event for: " << pair.first;
      return false;
    }
  }
  return CaptureCudaGraph(exec_stream);
}

// ── Binding introspection ────────────────────────────────────────────

std::vector<std::string> TensorRTEngine::GetInputNames() const {
  std::vector<std::string> names;
  for (const auto &pair : tensor_map_) if (pair.second->is_input()) names.push_back(pair.first);
  return names;
}

std::vector<std::string> TensorRTEngine::GetOutputNames() const {
  std::vector<std::string> names;
  for (const auto &pair : tensor_map_) if (pair.second->is_output()) names.push_back(pair.first);
  return names;
}

bool TensorRTEngine::HasInputBinding(const std::string &name) const {
  auto it = tensor_map_.find(name);
  return it != tensor_map_.end() && it->second->is_input();
}

bool TensorRTEngine::HasOutputBinding(const std::string &name) const {
  auto it = tensor_map_.find(name);
  return it != tensor_map_.end() && it->second->is_output();
}

bool TensorRTEngine::IsBindingInput(const std::string &name) const {
  auto it = tensor_map_.find(name);
  return it != tensor_map_.end() && it->second->is_input();
}

std::shared_ptr<Tensor> TensorRTEngine::GetBindingByName(const std::string &name) const {
  auto it = tensor_map_.find(name);
  return (it != tensor_map_.end()) ? it->second : nullptr;
}

// ── SetInputData ─────────────────────────────────────────────────────

bool TensorRTEngine::SetInputData(const std::string &name, const void *data, size_t size_bytes,
                                   const std::vector<int64_t> &dims, bool async) {
  const std::string nvtx_tag = engine_name_ + " input " + name;
  nvtx::NvtxRange nvtx_range(nvtx_tag.c_str());

  auto it = tensor_map_.find(name);
  if (it == tensor_map_.end() || !it->second->is_input()) {
    LOG(ERROR) << "Input tensor not found: " << name;
    return false;
  }

  nvinfer1::Dims d;
  d.nbDims = static_cast<int32_t>(dims.size());
  for (size_t j = 0; j < dims.size() && j < nvinfer1::Dims::MAX_DIMS; ++j) d.d[j] = dims[j];

  for (int32_t j = 0; j < d.nbDims; ++j) {
    if (d.d[j] <= 0) {
      LOG(ERROR) << "SetInputData: dimension must be > 0 for " << name;
      return false;
    }
  }

  // Validate input size
  size_t expected = 1;
  for (int32_t j = 0; j < d.nbDims; ++j) expected *= d.d[j];
  expected *= it->second->element_size();
  if (size_bytes != expected) {
    LOG(ERROR) << "Input size mismatch for " << name << ": expected " << expected << " bytes, got " << size_bytes;
    return false;
  }

  if (!it->second->Reshape(d)) {
    if (io_frozen_ || !it->second->Resize(d, it->second->dataType())) {
      LOG(ERROR) << "Failed to reshape input tensor " << name << " to requested dims";
      return false;
    }
    if (!context_->setTensorAddress(name.c_str(), it->second->mutable_data())) {
      LOG(ERROR) << "Failed to update tensor address after input resize for: " << name;
      return false;
    }
    ResetCudaGraph();
  }

  bool ok = false;
  if (async) {
    ok = it->second->CopyFromHostAsync(data, *stream_);
  } else {
    ok = it->second->CopyFromHost(data);
  }
  return ok && RecordInputReadyEvent(name);
}

// ── GetOutputData ────────────────────────────────────────────────────

bool TensorRTEngine::GetOutputData(const std::string &name, void *data, size_t size_bytes, bool async) {
  const std::string nvtx_tag = engine_name_ + " output " + name;
  nvtx::NvtxRange nvtx_range(nvtx_tag.c_str());

  if (!context_) {
    LOG(ERROR) << "Engine context not initialized";
    return false;
  }

  auto it = tensor_map_.find(name);
  if (it == tensor_map_.end() || !it->second->is_output()) {
    LOG(ERROR) << "Output tensor not found: " << name;
    return false;
  }

  // Dynamic-shape engines keep shape updates even after freezing because
  // shapes may still vary within the pre-allocated max profile bounds.
  if (!io_frozen_ || engine_has_dynamic_bindings_) {
    UpdateOutputShapes();
  }

  const auto &tensor = it->second;
  if (size_bytes != tensor->byte_size()) {
    LOG(ERROR) << "Output size mismatch for " << name << ": expected " << tensor->byte_size() << " bytes, got "
               << size_bytes;
    return false;
  }

  if (!WaitOutputReady(name, stream_->stream())) {
    return false;
  }

  if (async) {
    return tensor->CopyToHostAsync(data, *stream_);
  }
  return tensor->CopyToHost(data);
}

// ── Infer / InferAsync / Enqueue ─────────────────────────────────────

bool TensorRTEngine::Infer() {
  if (!InferAsync()) return false;
  return Synchronize();
}

bool TensorRTEngine::Enqueue() {
  const std::string nvtx_tag = engine_name_ + " enqueue";
  nvtx::NvtxRange nvtx_range(nvtx_tag.c_str());
  if (!engine_ || !context_) {
    CUERROR << "Engine not loaded";
    return false;
  }
  cudaStream_t exec_stream = GetCudaStream();
  if (exec_stream == nullptr) {
    LOG(ERROR) << "No valid CUDA stream for execution";
    return false;
  }
  CUDA_RETURN_VAL_IF(!context_->enqueueV3(exec_stream), false);
  return true;
}

bool TensorRTEngine::InferAsync() {
  if (!engine_ || !context_) {
    CUERROR << "Engine not loaded";
    return false;
  }

  cudaStream_t exec_stream = GetCudaStream();
  if (exec_stream == nullptr) {
    LOG(ERROR) << "No valid CUDA stream for execution";
    return false;
  }

  for (const auto &pair : tensor_map_) {
    if (!pair.second->is_input()) continue;
    auto event_it = input_ready_events_.find(pair.first);
    if (event_it != input_ready_events_.end() && !WaitEvent(exec_stream, event_it->second)) {
      LOG(ERROR) << "Failed to wait input ready event for: " << pair.first;
      return false;
    }
  }

  if (!io_frozen_ && !BindIOAddresses()) {
    return false;
  }

  bool launched = false;
  if (ShouldUseCudaGraph()) {
    if (!cuda_graph_captured_ || !CudaGraphShapesMatch()) {
      if (!CaptureCudaGraph(exec_stream)) {
        LOG(WARNING) << "CUDA Graph capture failed, falling back to enqueueV3";
        ResetCudaGraph();
        launched = Enqueue();
      } else {
        launched = LaunchCudaGraph(exec_stream);
      }
    } else {
      launched = LaunchCudaGraph(exec_stream);
    }
  } else {
    launched = Enqueue();
  }
  return launched && RecordOutputReadyEvents(exec_stream);
}

// ── Binding shape & profile ──────────────────────────────────────────

bool TensorRTEngine::SetBindingShape(const std::string &name, const nvinfer1::Dims &dims) {
  if (!context_) {
    LOG(ERROR) << "Engine context not initialized";
    return false;
  }
  if (io_frozen_ && !engine_has_dynamic_bindings_) {
    LOG(ERROR) << "Cannot change binding shape while I/O is frozen on a static engine: " << name;
    return false;
  }
  // Validate dims > 0
  for (int32_t j = 0; j < dims.nbDims; ++j) {
    if (dims.d[j] <= 0) {
      LOG(ERROR) << "SetBindingShape: dimension must be > 0, got " << dims.d[j] << " at axis " << j
                 << " for binding " << name;
      return false;
    }
  }
  bool ok = context_->setInputShape(name.c_str(), dims);
  if (ok && io_frozen_) {
    ResetCudaGraph();
  }
  return ok;
}

bool TensorRTEngine::SetOptimizationProfile(int32_t profile_index) {
  if (!context_) return false;
  if (io_frozen_) {
    LOG(ERROR) << "Cannot switch optimization profile while I/O is frozen";
    return false;
  }
  ResetCudaGraph();
  return context_->setOptimizationProfileAsync(profile_index, stream_->stream());
}

bool TensorRTEngine::UpdateOutputShapes() {
  if (!engine_ || !context_) {
    LOG(ERROR) << "Engine not loaded";
    return false;
  }

  for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
    const std::string name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name.c_str()) != nvinfer1::TensorIOMode::kOUTPUT) continue;

    nvinfer1::Dims dims = context_->getTensorShape(name.c_str());
    if (dims.nbDims < 0) {
      LOG(ERROR) << "Failed to get output shape for: " << name;
      return false;
    }
    if (HasUnresolvedDims(dims)) continue;

    auto it = tensor_map_.find(name);
    if (it == tensor_map_.end()) continue;

    auto &tensor = it->second;
    size_t new_num_elements = 1;
    for (int j = 0; j < dims.nbDims; ++j) {
      if (dims.d[j] <= 0) {
        LOG(ERROR) << "UpdateOutputShapes: output dimension must be > 0, got " << dims.d[j]
                   << " at axis " << j << " for binding " << name;
        return false;
      }
      new_num_elements *= static_cast<size_t>(dims.d[j]);
    }
    size_t new_byte_size = new_num_elements * tensor->element_size();

    if (new_byte_size > tensor->capacity_bytes()) {
      if (io_frozen_) {
        LOG(ERROR) << "Output shape changed requiring reallocation while I/O is frozen: " << name;
        return false;
      }
      if (!tensor->Resize(new_byte_size)) {
        LOG(ERROR) << "Failed to resize output tensor: " << name;
        return false;
      }
      if (!context_->setTensorAddress(name.c_str(), tensor->mutable_data())) {
        LOG(ERROR) << "Failed to update tensor address after resize for: " << name;
        return false;
      }
    } else {
      // Reshape in-place without reallocation
      if (!tensor->Reshape(dims)) {
        LOG(ERROR) << "Failed to reshape output tensor: " << name;
        return false;
      }
    }
  }
  return true;
}

// ── Binding address ──────────────────────────────────────────────────

bool TensorRTEngine::SetBindingAddress(const std::string &name, void *device_ptr) {
  if (!context_) {
    LOG(ERROR) << "Engine context not initialized";
    return false;
  }
  if (io_frozen_) {
    LOG(ERROR) << "Cannot change binding address while I/O is frozen: " << name;
    return false;
  }
  CUDA_RETURN_VAL_IF(!context_->setTensorAddress(name.c_str(), device_ptr), false);
  user_binding_addresses_[name] = device_ptr;
  ResetCudaGraph();
  return true;
}

void *TensorRTEngine::GetBindingAddress(const std::string &name) const {
  if (!context_) {
    LOG(ERROR) << "Engine context not initialized";
    return nullptr;
  }
  return const_cast<void *>(context_->getTensorAddress(name.c_str()));
}

// ── FreezeIO / Execute ───────────────────────────────────────────────

bool TensorRTEngine::FreezeIO() {
  if (!engine_ || !context_) return false;
  if (!config_.enable_persistent_io_buffers) {
    LOG(ERROR) << "Cannot freeze I/O when persistent I/O buffer fast paths are disabled";
    return false;
  }
  if (!io_frozen_) {
    if (!UpdateOutputShapes()) return false;
    if (!BindIOAddresses()) return false;
    ResetCudaGraph();
    io_frozen_ = true;
  }
  return io_frozen_;
}

bool TensorRTEngine::Execute() {
  return Infer();
}

bool TensorRTEngine::Execute(const std::unordered_map<std::string, void *> &inputs,
                              const std::unordered_map<std::string, void *> &outputs) {
  if (!engine_ || !context_) return false;

  auto dims_to_vector = [](const nvinfer1::Dims &dims) {
    std::vector<int64_t> vec;
    vec.reserve(dims.nbDims);
    for (int32_t i = 0; i < dims.nbDims; ++i) vec.push_back(dims.d[i]);
    return vec;
  };

  // Inputs: host pointers go through SetInputData; device pointers are bound directly.
  for (const auto &[name, ptr] : inputs) {
    if (!IsDeviceAccessiblePointer(ptr)) {
      nvinfer1::Dims dims = GetTensorShape(name);
      if (HasUnresolvedDims(dims) || !HasConcreteDims(dims)) {
        LOG(ERROR) << "Execute: input shape not set for " << name;
        return false;
      }
      size_t num_elements = 1;
      for (int i = 0; i < dims.nbDims; ++i) num_elements *= dims.d[i];
      auto tensor = GetBindingByName(name);
      if (!tensor) {
        LOG(ERROR) << "Execute: input tensor not found: " << name;
        return false;
      }
      size_t size_bytes = num_elements * tensor->element_size();
      if (!SetInputData(name, ptr, size_bytes, dims_to_vector(dims), false)) return false;
    } else {
      if (!SetBindingAddress(name, ptr)) return false;
    }
  }

  // Device outputs: bind directly.
  for (const auto &[name, ptr] : outputs) {
    if (IsDeviceAccessiblePointer(ptr)) {
      if (!SetBindingAddress(name, ptr)) return false;
    }
  }

  if (!Infer()) return false;

  // Host outputs: copy back via GetOutputData.
  for (const auto &[name, ptr] : outputs) {
    if (!IsDeviceAccessiblePointer(ptr)) {
      auto tensor = GetBindingByName(name);
      if (!tensor) {
        LOG(ERROR) << "Execute: output tensor not found: " << name;
        return false;
      }
      nvinfer1::Dims dims = GetTensorShape(name);
      if (HasUnresolvedDims(dims) || !HasConcreteDims(dims)) {
        LOG(ERROR) << "Execute: output shape not resolved for " << name;
        return false;
      }
      size_t num_elements = 1;
      for (int i = 0; i < dims.nbDims; ++i) num_elements *= dims.d[i];
      size_t size_bytes = num_elements * tensor->element_size();
      if (!GetOutputData(name, ptr, size_bytes, false)) return false;
    }
  }
  return true;
}

// ── LLM-oriented extensions ──────────────────────────────────────────

int32_t TensorRTEngine::GetNbIOTensors() const {
  if (!engine_) return 0;
  return engine_->getNbIOTensors();
}

std::string TensorRTEngine::GetIOTensorName(int32_t index) const {
  if (!engine_) return "";
  const int32_t nb = engine_->getNbIOTensors();
  if (index < 0 || index >= nb) return "";
  return engine_->getIOTensorName(index);
}

nvinfer1::DataType TensorRTEngine::GetTensorDataType(const std::string &name) const {
  if (!engine_) return nvinfer1::DataType::kFLOAT;
  return engine_->getTensorDataType(name.c_str());
}

nvinfer1::Dims TensorRTEngine::GetTensorShape(const std::string &name) const {
  if (!context_) {
    nvinfer1::Dims invalid;
    invalid.nbDims = -1;
    return invalid;
  }
  return context_->getTensorShape(name.c_str());
}

nvinfer1::Dims TensorRTEngine::GetEngineTensorShape(const std::string &name) const {
  if (!engine_) {
    nvinfer1::Dims invalid;
    invalid.nbDims = -1;
    return invalid;
  }
  return engine_->getTensorShape(name.c_str());
}

nvinfer1::Dims TensorRTEngine::GetProfileShape(const std::string &name, int32_t profile_index,
                                                nvinfer1::OptProfileSelector selector) const {
  nvinfer1::Dims invalid;
  invalid.nbDims = -1;
  if (!engine_) return invalid;
  return engine_->getProfileShape(name.c_str(), profile_index, selector);
}

int64_t TensorRTEngine::GetDeviceMemorySize() const {
  if (!context_) return 0;
  return engine_->getDeviceMemorySizeV2();
}

bool TensorRTEngine::SetDeviceMemory(void *memory, int64_t size) {
  if (!context_) {
    LOG(ERROR) << "Engine context not initialized";
    return false;
  }
  context_->setDeviceMemoryV2(memory, size);
  return true;
}

bool TensorRTEngine::IsDynamicBinding(const std::string &name) const {
  if (!engine_) return false;
  auto shape = engine_->getTensorShape(name.c_str());
  for (int j = 0; j < shape.nbDims; ++j) if (shape.d[j] < 0) return true;
  return false;
}

// ── Helpers ──────────────────────────────────────────────────────────

bool TensorRTEngine::LoadPluginLibrary(const std::string &plugin_path) {
  plugin_handle_ = dlopen(plugin_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!plugin_handle_) {
    LOG(ERROR) << "Failed to load plugin library: " << plugin_path;
    return false;
  }
  return true;
}

bool TensorRTEngine::RecordInputReadyEvent(const std::string &name) {
  auto event = std::make_shared<CudaEvent>(CudaEvent::Flag::DISABLE_TIMING);
  if (!event->Record(stream_->stream())) {
    LOG(ERROR) << "Failed to record input ready event for: " << name;
    return false;
  }
  input_ready_events_[name] = event;
  return true;
}

bool TensorRTEngine::RecordOutputReadyEvents(cudaStream_t stream) {
  CUDA_RETURN_VAL_IF(stream == nullptr, false);
  for (const auto &pair : tensor_map_) {
    if (!pair.second->is_output()) continue;

    auto event = std::make_shared<CudaEvent>(CudaEvent::Flag::DISABLE_TIMING);
    if (!event->Record(stream)) {
      LOG(ERROR) << "Failed to record output ready event for: " << pair.first;
      return false;
    }
    output_ready_events_[pair.first] = event;
  }
  return true;
}

bool TensorRTEngine::WaitEvent(cudaStream_t stream, const std::shared_ptr<CudaEvent> &event) const {
  CUDA_RETURN_VAL_IF(stream == nullptr || event == nullptr, false);
  CUDART_RETURN_CALL(cudaStreamWaitEvent(stream, event->event(), 0));
  return true;
}

bool TensorRTEngine::BindIOAddresses() {
  if (!context_) {
    LOG(ERROR) << "Engine context not initialized";
    return false;
  }

  for (auto &pair : tensor_map_) {
    void *addr = pair.second->mutable_data();
    auto user_it = user_binding_addresses_.find(pair.first);
    if (user_it != user_binding_addresses_.end()) {
      addr = user_it->second;
    }
    if (!context_->setTensorAddress(pair.first.c_str(), addr)) {
      LOG(ERROR) << "Failed to set tensor address for: " << pair.first;
      return false;
    }
  }
  return true;
}

bool TensorRTEngine::CaptureCudaGraph(cudaStream_t stream) {
  if (!ShouldUseCudaGraph()) {
    LOG(ERROR) << "CUDA Graph capture requires enabled CUDA Graph and frozen I/O";
    return false;
  }
  CUDA_RETURN_VAL_IF(stream == nullptr, false);

  ResetCudaGraph();

  cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
  if (err != cudaSuccess) {
    LOG(ERROR) << "cudaStreamBeginCapture failed: " << cudaGetErrorString(err);
    cudaGetLastError();
    return false;
  }

  bool enqueue_ok = context_->enqueueV3(stream);
  cudaGraph_t captured_graph = nullptr;
  err = cudaStreamEndCapture(stream, &captured_graph);
  if (!enqueue_ok || err != cudaSuccess || captured_graph == nullptr) {
    if (err != cudaSuccess) {
      LOG(ERROR) << "cudaStreamEndCapture failed: " << cudaGetErrorString(err);
      cudaGetLastError();
    } else if (!enqueue_ok) {
      LOG(ERROR) << "TensorRT enqueueV3 failed during CUDA Graph capture";
    } else {
      LOG(ERROR) << "CUDA Graph capture returned a null graph";
    }
    if (captured_graph != nullptr) {
      cudaGraphDestroy(captured_graph);
    }
    return false;
  }

  cudaGraphExec_t captured_exec = nullptr;
  err = cudaGraphInstantiate(&captured_exec, captured_graph, 0);
  if (err != cudaSuccess) {
    LOG(ERROR) << "cudaGraphInstantiate failed: " << cudaGetErrorString(err);
    cudaGetLastError();
    cudaGraphDestroy(captured_graph);
    return false;
  }

  cuda_graph_ = captured_graph;
  cuda_graph_exec_ = captured_exec;
  cuda_graph_captured_ = true;
  SnapshotCudaGraphShapes();
  return true;
}

bool TensorRTEngine::LaunchCudaGraph(cudaStream_t stream) {
  CUDA_RETURN_VAL_IF(stream == nullptr || cuda_graph_exec_ == nullptr, false);
  CUDART_RETURN_CALL(cudaGraphLaunch(cuda_graph_exec_, stream));
  return true;
}

void TensorRTEngine::ResetCudaGraph() {
  if (cuda_graph_exec_ != nullptr) {
    cudaGraphExecDestroy(cuda_graph_exec_);
    cuda_graph_exec_ = nullptr;
  }
  if (cuda_graph_ != nullptr) {
    cudaGraphDestroy(cuda_graph_);
    cuda_graph_ = nullptr;
  }
  cuda_graph_tensor_shapes_.clear();
  cuda_graph_captured_ = false;
}

bool TensorRTEngine::ShouldUseCudaGraph() const {
  return cuda_graph_enabled_ && io_frozen_ && engine_ != nullptr && context_ != nullptr;
}

bool TensorRTEngine::CudaGraphShapesMatch() const {
  if (!cuda_graph_captured_) return false;
  for (const auto &pair : tensor_map_) {
    auto shape_it = cuda_graph_tensor_shapes_.find(pair.first);
    if (shape_it == cuda_graph_tensor_shapes_.end()) return false;

    nvinfer1::Dims current = pair.second->is_input() ? context_->getTensorShape(pair.first.c_str())
                                                     : pair.second->dims();
    if (!SameDims(current, shape_it->second)) return false;
  }
  return true;
}

void TensorRTEngine::SnapshotCudaGraphShapes() {
  cuda_graph_tensor_shapes_.clear();
  for (const auto &pair : tensor_map_) {
    cuda_graph_tensor_shapes_[pair.first] =
        pair.second->is_input() ? context_->getTensorShape(pair.first.c_str()) : pair.second->dims();
  }
}

bool TensorRTEngine::SameDims(const nvinfer1::Dims &lhs, const nvinfer1::Dims &rhs) const {
  if (lhs.nbDims != rhs.nbDims) return false;
  for (int32_t i = 0; i < lhs.nbDims; ++i) {
    if (lhs.d[i] != rhs.d[i]) return false;
  }
  return true;
}

bool TensorRTEngine::HasUnresolvedDims(const nvinfer1::Dims &dims) const {
  for (int32_t j = 0; j < dims.nbDims; ++j) {
    if (dims.d[j] == -1) return true;
  }
  return false;
}

bool TensorRTEngine::HasConcreteDims(const nvinfer1::Dims &dims) const {
  if (dims.nbDims < 0) return false;
  for (int32_t i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0) return false;
  }
  return true;
}

bool TensorRTEngine::IsDeviceAccessiblePointer(const void *ptr) const {
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
  if (err != cudaSuccess) {
    cudaGetLastError();  // clear error
    return false;
  }
#if CUDART_VERSION >= 10000
  return (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged);
#else
  return (attr.memoryType == cudaMemoryTypeDevice);
#endif
}

}  // namespace tensorrt_engine
