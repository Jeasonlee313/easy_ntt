#ifndef TENSORRT_ENGINE_TENSORRT_UTILITY_HPP_
#define TENSORRT_ENGINE_TENSORRT_UTILITY_HPP_

#include <NvInfer.h>

#include <string>

namespace tensorrt_engine {
namespace utility {

// String representation of TensorRT tensor data type (shared by TensorRTEngine and gr00t TrtEngine).
std::string dataTypeToString(nvinfer1::DataType type);

// Element size in bytes (uses float for partial-byte types such as INT4 in TRT).
float getDataTypeSize(nvinfer1::DataType type);

}  // namespace utility
}  // namespace tensorrt_engine

#endif  // TENSORRT_ENGINE_TENSORRT_UTILITY_HPP_
