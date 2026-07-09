#ifndef TENSORRT_ENGINE_TENSORRT_ENGINE_HPP_
#define TENSORRT_ENGINE_TENSORRT_ENGINE_HPP_

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <functional>
#include <future>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "driver/cuda_event.h"
#include "driver/cuda_stream.h"
#include "tensorrt_engine/tensor.h"
#include "tensorrt_engine/tensorrt_utility.hpp"
#include "tensorrt_engine/trt_logger.h"

namespace tensorrt_engine {
using driver::event::CudaEvent;
using driver::stream::CudaStream;
using tensorrt_engine::logger::Logger;
using tensorrt_engine::tensor::Tensor;

class TensorRTEngine {
 public:
  // Updated constructor that takes engine path, optional plugin path, and optional CUDA stream
  TensorRTEngine(const std::string &engine_path, const std::string &plugin_path = "");
  ~TensorRTEngine();

  std::shared_ptr<Tensor> GetBindingByName(const std::string &name) const;

  const std::unordered_map<std::string, std::shared_ptr<Tensor>> &GetTensorMap() const { return tensor_map_; }

  // Set input data
  bool SetInputData(const std::string &name, const void *data, size_t size_bytes, const std::vector<int64_t> &dims,
                    bool async);

  // Get output data
  bool GetOutputData(const std::string &name, void *data, size_t size_bytes, bool async);

  // Run inference (synchronous). Deprecated: use InferAsync() plus event control
  // or explicit Synchronize() instead.
  [[deprecated("Use InferAsync() with tensor event control or explicit Synchronize() instead.")]]
  bool Infer();

  // Run inference (asynchronous)
  bool InferAsync();

  // Check if engine is loaded
  bool IsLoaded() const;

  // ---------------------------------------------------------------------------
  // Stream control
  // ---------------------------------------------------------------------------

  //! @brief Replace the internal CUDA stream with an external one.
  bool SetStream(cudaStream_t stream);

  //! @brief Get the underlying cudaStream_t handle.
  cudaStream_t GetCudaStream() const;

  //! @brief Synchronize the execution stream.
  bool Synchronize() const;

  // ---------------------------------------------------------------------------
  // Tensor event control
  // ---------------------------------------------------------------------------

  //! @brief Get the CUDA event recorded after an input tensor copy/update.
  cudaEvent_t GetInputReadyEvent(const std::string &name) const;

  //! @brief Get the CUDA event recorded after inference has produced an output tensor.
  cudaEvent_t GetOutputReadyEvent(const std::string &name) const;

  //! @brief Make a stream wait for the selected input-ready event.
  bool WaitInputReady(const std::string &name, cudaStream_t stream) const;

  //! @brief Make a stream wait for the selected output-ready event.
  bool WaitOutputReady(const std::string &name, cudaStream_t stream) const;

  //! @brief Host-side synchronization on a selected input-ready event.
  bool SynchronizeInput(const std::string &name) const;

  //! @brief Host-side synchronization on a selected output-ready event.
  bool SynchronizeOutput(const std::string &name) const;

  // ---------------------------------------------------------------------------
  // Binding query helpers
  // ---------------------------------------------------------------------------

  //! @brief Get all input tensor names.
  std::vector<std::string> GetInputNames() const;

  //! @brief Get all output tensor names.
  std::vector<std::string> GetOutputNames() const;

  //! @brief Check if a binding exists and is an input.
  bool HasInputBinding(const std::string &name) const;

  //! @brief Check if a binding exists and is an output.
  bool HasOutputBinding(const std::string &name) const;

  //! @brief Check if a binding is an input.
  bool IsBindingInput(const std::string &name) const;

  // ---------------------------------------------------------------------------
  // LLM-oriented extensions
  // ---------------------------------------------------------------------------

  //! @brief Set the shape of an input binding and resize its Tensor if needed.
  //!        Validates against engine's min/opt/max profile shapes.
  //!        After FreezeIO on a static-shape engine this is rejected.
  //!        After FreezeIO on a dynamic-shape engine it is still allowed within
  //!        the max profile bounds because memory is allocated at max profile.
  bool SetBindingShape(const std::string &name, const nvinfer1::Dims &dims);

  //! @brief Switch optimization profile (e.g., prefill=0, generation=1).
  bool SetOptimizationProfile(int32_t profile_index);

  //! @brief Directly bind a device pointer to a binding (bypasses internal Tensor).
  //!        Useful for KV-cache or externally-managed memory.
  //!        Rejected after FreezeIO because addresses are locked.
  bool SetBindingAddress(const std::string &name, void *device_ptr);

  //! @brief Get the current device pointer bound to a binding.
  void *GetBindingAddress(const std::string &name) const;

  //! @brief Check if a binding has dynamic dimensions.
  bool IsDynamicBinding(const std::string &name) const;

  // ---------------------------------------------------------------------------
  // Extended getters for LLM Engine Runner (Module 7)
  // ---------------------------------------------------------------------------

  //! @brief Get the underlying TensorRT engine (for advanced operations).
  nvinfer1::ICudaEngine *GetEngine() const { return engine_.get(); }

  //! @brief Get the underlying execution context.
  nvinfer1::IExecutionContext *GetContext() const { return context_.get(); }

  //! @brief Get number of I/O tensors (TRT 8.6+ API).
  int32_t GetNbIOTensors() const;

  //! @brief Get I/O tensor name by index (TRT 8.6+ API).
  std::string GetIOTensorName(int32_t index) const;

  //! @brief Get tensor data type by name.
  nvinfer1::DataType GetTensorDataType(const std::string &name) const;

  //! @brief Get current tensor shape from execution context (reflects actual shape after dynamic inference).
  nvinfer1::Dims GetTensorShape(const std::string &name) const;

  //! @brief Get engine-defined static tensor shape (may contain -1 for dynamic dimensions).
  nvinfer1::Dims GetEngineTensorShape(const std::string &name) const;

  //! @brief Get profile shape for a binding.
  nvinfer1::Dims GetProfileShape(const std::string &name, int32_t profile_index,
                                 nvinfer1::OptProfileSelector selector) const;

  //! @brief Get required device memory size for the execution context.
  int64_t GetDeviceMemorySize() const;

  //! @brief Set shared device memory for the execution context.
  bool SetDeviceMemory(void *memory, int64_t size);

  // ---------------------------------------------------------------------------
  // Dynamic-shape & static fast path
  // ---------------------------------------------------------------------------

  //! @brief Freeze I/O for fast path.
  //!        - Static-shape engine: locks both shapes and addresses.
  //!        - Dynamic-shape engine: locks addresses only; shapes may still vary
  //!          within the max profile bounds (memory is already allocated at max).
  //!        After freezing, InferAsync skips redundant setTensorAddress calls.
  bool FreezeIO();

  //! @brief Enable/disable static execution fast path.
  void EnableStaticExecution(bool enable) { static_execution_enabled_ = enable; }

  //! @brief Check if I/O is frozen (addresses locked).
  bool IsIOFrozen() const { return io_frozen_; }

  //! @brief One-shot execute using internal tensor_map_.
  //!        Caller must have already set input shapes (SetBindingShape) and data.
  bool Execute();

  //! @brief One-shot execute with explicit host/device pointers.
  //!        Automatically detects pointer type and skips H2D for device pointers.
  //!        Input shapes must already be set via SetBindingShape for dynamic engines.
  bool Execute(const std::unordered_map<std::string, void *> &inputs,
               const std::unordered_map<std::string, void *> &outputs);

 private:
  // TensorRT objects
  Logger logger_;
  std::shared_ptr<nvinfer1::IRuntime> runtime_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  std::string engine_name_;

  // CUDA stream for async operations
  std::shared_ptr<CudaStream> stream_ = std::make_shared<CudaStream>(CudaStream::Flag::NON_BLOCKING);

  // External stream injected via SetStream (non-owning; caller guarantees lifetime).
  cudaStream_t external_stream_ = nullptr;

  // Binding information
  std::unordered_map<std::string, std::shared_ptr<Tensor>> tensor_map_;

  // User-overridden binding addresses (set via SetBindingAddress).
  std::unordered_map<std::string, void *> user_binding_addresses_;

  // Tensor readiness events for cross-stream input/output control.
  std::unordered_map<std::string, std::shared_ptr<CudaEvent>> input_ready_events_;
  std::unordered_map<std::string, std::shared_ptr<CudaEvent>> output_ready_events_;

  // Plugin library handle
  void *plugin_handle_;

  // Auxiliary streams for multi-stream engines
  std::vector<cudaStream_t> aux_streams_;

  // Fast-path flags
  bool io_frozen_ = false;
  bool static_execution_enabled_ = false;

  // True if the loaded engine has at least one dynamic binding.
  bool engine_has_dynamic_bindings_ = false;

  // Clean up resources
  void Cleanup();

  void AllocateIOBindingsMemory();

  //! @brief Update output Tensor shapes after enqueueV3 (for dynamic outputs).
  //!        Resizes Tensors if capacity is insufficient.
  bool UpdateOutputShapes();

  // Load plugin library
  bool LoadPluginLibrary(const std::string &plugin_path);

  // Core enqueue (no address binding)
  bool Enqueue();

  // Helpers
  bool RecordInputReadyEvent(const std::string &name);
  bool RecordOutputReadyEvents(cudaStream_t stream);
  bool WaitEvent(cudaStream_t stream, const std::shared_ptr<CudaEvent> &event) const;
  bool IsDeviceAccessiblePointer(const void *ptr) const;
  bool HasUnresolvedDims(const nvinfer1::Dims &dims) const;
  bool HasConcreteDims(const nvinfer1::Dims &dims) const;
};
}  // namespace tensorrt_engine

#endif  // TENSORRT_ENGINE_TENSORRT_ENGINE_HPP_
