//
// Created by jeason on 2/10/26.
//

#include "plugins/custom_plugins.h"

#include "plugins/bev_pool_v2/trt_bev_pool.h"

namespace nvinfer1 {
namespace plugin {
void CustomPlugins::RegisterAll() { REGISTER_TENSORRT_PLUGIN(TRTBEVPoolV2Creator); }
}  // namespace plugin
}  // namespace nvinfer1
