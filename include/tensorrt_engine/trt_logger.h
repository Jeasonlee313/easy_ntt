//
// Created by jeason on 2/24/26.
//

#ifndef TENSORRT_ENGINE_TRT_LOGGER_H
#define TENSORRT_ENGINE_TRT_LOGGER_H

#include <NvInfer.h>

#include <iostream>

namespace tensorrt_engine {
namespace logger {

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char *msg) noexcept override;
};

}  // namespace logger
}  // namespace tensorrt_engine

#endif  // TENSORRT_ENGINE_TRT_LOGGER_H
