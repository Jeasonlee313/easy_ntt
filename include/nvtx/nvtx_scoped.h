#pragma once

namespace nvtx {

// Global enable for inference NVTX ranges (false during warm-up pauses).
bool NvtxRuntimeInferenceEnabled();
void SetNvtxRuntimeInferenceEnabled(bool enabled);

// Pushes nvtxRangePushA(name) when NvtxRuntimeInferenceEnabled().
struct NvtxRange {
  explicit NvtxRange(const char* name);
  ~NvtxRange();
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;

 private:
  bool pushed_;
};

// Pushes only when condition && NvtxRuntimeInferenceEnabled().
struct NvtxRangeIf {
  NvtxRangeIf(bool condition, const char* name);
  ~NvtxRangeIf();
  NvtxRangeIf(const NvtxRangeIf&) = delete;
  NvtxRangeIf& operator=(const NvtxRangeIf&) = delete;

 private:
  bool pushed_;
};

// RAII: disable inference NVTX for nested calls (engine warm-up).
struct NvtxWarmupPause {
  NvtxWarmupPause();
  ~NvtxWarmupPause();
  NvtxWarmupPause(const NvtxWarmupPause&) = delete;
  NvtxWarmupPause& operator=(const NvtxWarmupPause&) = delete;

 private:
  bool previous_;
};

}  // namespace nvtx
