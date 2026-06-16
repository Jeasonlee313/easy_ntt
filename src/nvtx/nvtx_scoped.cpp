#include "nvtx/nvtx_scoped.h"

#ifdef NVTX_AVAILABLE
#include <nvToolsExt.h>
#endif

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace nvtx {
namespace {

std::atomic<bool> g_nvtx_runtime_inference_enabled{true};

// GR00T_DISABLE_NVTX=1：推理路径跳过 nvtxRangePush/Pop，板上可减少零点几毫秒量级开销。
bool EnvDisablesNvtx() {
  static const bool disabled = []() {
    const char* v = std::getenv("GR00T_DISABLE_NVTX");
    return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0 && std::strcmp(v, "false") != 0 &&
           std::strcmp(v, "no") != 0 && std::strcmp(v, "FALSE") != 0 && std::strcmp(v, "No") != 0;
  }();
  return disabled;
}

}  // namespace

bool NvtxRuntimeInferenceEnabled() {
  if (EnvDisablesNvtx()) {
    return false;
  }
  return g_nvtx_runtime_inference_enabled.load(std::memory_order_relaxed);
}

void SetNvtxRuntimeInferenceEnabled(bool enabled) {
  g_nvtx_runtime_inference_enabled.store(enabled, std::memory_order_relaxed);
}

NvtxRange::NvtxRange(const char* name) : pushed_(NvtxRuntimeInferenceEnabled() && name != nullptr) {
#ifdef NVTX_AVAILABLE
  if (pushed_) {
    nvtxRangePushA(name);
  }
#endif
}

NvtxRange::~NvtxRange() {
#ifdef NVTX_AVAILABLE
  if (pushed_) {
    nvtxRangePop();
  }
#endif
}

NvtxRangeIf::NvtxRangeIf(bool condition, const char* name)
    : pushed_(condition && NvtxRuntimeInferenceEnabled() && name != nullptr) {
#ifdef NVTX_AVAILABLE
  if (pushed_) {
    nvtxRangePushA(name);
  }
#endif
}

NvtxRangeIf::~NvtxRangeIf() {
#ifdef NVTX_AVAILABLE
  if (pushed_) {
    nvtxRangePop();
  }
#endif
}

NvtxWarmupPause::NvtxWarmupPause() : previous_(NvtxRuntimeInferenceEnabled()) { SetNvtxRuntimeInferenceEnabled(false); }

NvtxWarmupPause::~NvtxWarmupPause() { SetNvtxRuntimeInferenceEnabled(previous_); }

}  // namespace nvtx
