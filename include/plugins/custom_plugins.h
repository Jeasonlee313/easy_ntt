//
// Created by jeason on 2/10/26.
//

#ifndef TENSORRT_ENGINE_CUSTOM_PLUGINS_H
#define TENSORRT_ENGINE_CUSTOM_PLUGINS_H

#include <NvInfer.h>
#include <NvInferPlugin.h>

namespace nvinfer1 {
namespace plugin {

class CustomPlugins {
 public:
  static void RegisterAll();
};

}  // namespace plugin
}  // namespace nvinfer1

#endif  // TENSORRT_ENGINE_CUSTOM_PLUGINS_H
