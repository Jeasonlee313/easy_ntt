//
// Created by jeason on 2/24/26.
//

#include "tensorrt_engine/trt_logger.h"

#include "driver/cuda_macros.h"

namespace tensorrt_engine {
namespace logger {
// Logger implementation
void Logger::log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept {
  switch (severity) {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
    case nvinfer1::ILogger::Severity::kERROR:
      CUERROR << "TensorRT ERROR: " << msg << std::endl;
      break;
    case nvinfer1::ILogger::Severity::kWARNING:
      CUWARN << "TensorRT WARNING: " << msg << std::endl;
      break;
    case nvinfer1::ILogger::Severity::kINFO:
      CUINFO << "TensorRT INFO: " << msg;
      break;
    default:
      break;
  }
}

}  // namespace logger
}  // namespace tensorrt_engine
