//
// Created by jeason on 1/28/26.
//

#ifndef TENSORRT_ENGINE_CUDA_MEMORY_TYPE_H
#define TENSORRT_ENGINE_CUDA_MEMORY_TYPE_H

enum class CudaMemoryType {
  INVALID = 0,
  DEVICE,
  PINNED,
  PINNED_WC,
};

#endif  // TENSORRT_ENGINE_CUDA_MEMORY_TYPE_H
