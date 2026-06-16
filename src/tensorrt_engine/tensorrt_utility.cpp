#include "tensorrt_engine/tensorrt_utility.hpp"

#include <cmath>
#include <stdexcept>

namespace tensorrt_engine {
namespace utility {

std::string dataTypeToString(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
      return "FLOAT";
    case nvinfer1::DataType::kHALF:
      return "HALF";
    case nvinfer1::DataType::kINT8:
      return "INT8";
    case nvinfer1::DataType::kINT32:
      return "INT32";
    case nvinfer1::DataType::kBOOL:
      return "BOOL";
    case nvinfer1::DataType::kUINT8:
      return "UINT8";
    case nvinfer1::DataType::kFP8:
      return "FP8";
    case nvinfer1::DataType::kBF16:
      return "BF16";
    case nvinfer1::DataType::kINT64:
      return "INT64";
    default:
      return "UNKNOWN";
  }
}

float getDataTypeSize(nvinfer1::DataType type) {
  using nvinfer1::DataType;
  switch (type) {
    case DataType::kFLOAT:
      return 4;  // float32
    case DataType::kHALF:
      return 2;  // float16
    case DataType::kINT8:
      return 1;  // int8
    case DataType::kINT32:
      return 4;  // int32
    case DataType::kBOOL:
      return 1;  // bool
    case DataType::kUINT8:
      return 1;  // uint8 (TRT 8.6+)
    case DataType::kFP8:
      return 1;  // fp8
    case DataType::kBF16:
      return 2;  // bfloat16
    case DataType::kINT64:
      return 8;  // int64 (TRT 8.6+)
    case DataType::kINT4:
      return 0.5f;  // int4 (TRT 8.6+)
    default:
      throw std::runtime_error("Unknown or unsupported nvinfer1::DataType.");
  }
}

}  // namespace utility
}  // namespace tensorrt_engine
