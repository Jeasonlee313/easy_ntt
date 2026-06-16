//
// Created by jeason on 1/27/26.
//

#ifndef TENSORRT_ENGINE_CUDA_MACROS_H
#define TENSORRT_ENGINE_CUDA_MACROS_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>

#define CUINFO LOG(INFO)
#define CUWARN LOG(WARNING)
#define CUERROR LOG(ERROR)
#define CUFATAL LOG(FATAL)

#define CUDA_RETURN_VAL_IF(condition, ret)                 \
  do {                                                     \
    bool _cuda_macros_value = condition;                   \
    if (_cuda_macros_value) {                              \
      LOG(WARNING) << #condition << " is true, return..."; \
      return ret;                                          \
    }                                                      \
  } while (0);

#define CUDA_RETURN_VAL_IF_QUIET(condition, ret) \
  do {                                           \
    bool _cuda_macros_value = condition;         \
    if (_cuda_macros_value) {                    \
      return ret;                                \
    }                                            \
  } while (0);

#define CUDA_RETURN_IF_QUIET(condition)  \
  do {                                   \
    bool _cuda_macros_value = condition; \
    if (_cuda_macros_value) {            \
      return;                            \
    }                                    \
  } while (0);

#define CUDA_CHECK(condition) \
  if ((condition) == false) LOG(FATAL) << "CUDA Check " << #condition

#define CUDA_CHECK_NOTNULL(condition) \
  if ((condition) == nullptr) LOG(FATAL) << #condition + std::string(" == nullptr")

#define CUDART_CALL(status)                                       \
  do {                                                            \
    cudaError_t err = status;                                     \
    if (err != cudaSuccess) {                                     \
      LOG(ERROR) << "CUDA API Error " << cudaGetErrorString(err); \
      cudaGetLastError();                                         \
    }                                                             \
  } while (0)

#define CUDART_RETURN_CALL(status)                                \
  do {                                                            \
    cudaError_t err = status;                                     \
    if (err != cudaSuccess) {                                     \
      LOG(ERROR) << "CUDA API Error " << cudaGetErrorString(err); \
      cudaGetLastError();                                         \
      return false;                                               \
    }                                                             \
  } while (0)

#define CUDART_RETURN_IF_ERROR(err_code)                               \
  do {                                                                 \
    if (err_code != cudaSuccess) {                                     \
      LOG(ERROR) << "CUDA API Error " << cudaGetErrorString(err_code); \
      cudaGetLastError();                                              \
      return false;                                                    \
    }                                                                  \
  } while (0)

#define CUDART_RETURN_VAL_IF_FAIL(status, ret)                    \
  do {                                                            \
    cudaError_t err = status;                                     \
    if (err != cudaSuccess) {                                     \
      LOG(ERROR) << "CUDA API Error " << cudaGetErrorString(err); \
      cudaGetLastError();                                         \
      return ret;                                                 \
    }                                                             \
  } while (0)

#define CUDART_CHECK()                                                \
  do {                                                                \
    cudaError_t err = cudaGetLastError();                             \
    if (err != cudaSuccess) {                                         \
      LOG(ERROR) << "CUDA Kernel Error: " << cudaGetErrorString(err); \
    }                                                                 \
  } while (0)

#define CUDART_RETURN_CHECK()                                         \
  do {                                                                \
    cudaError_t err = cudaGetLastError();                             \
    if (err != cudaSuccess) {                                         \
      LOG(ERROR) << "CUDA Kernel Error: " << cudaGetErrorString(err); \
      return false;                                                   \
    }                                                                 \
  } while (0)

#define CUDA_CALL(status)                                \
  do {                                                   \
    if ((status) != CUDA_SUCCESS) {                      \
      const char *err_str;                               \
      cuGetErrorString((status), &err_str);              \
      LOG(ERROR) << "CUDA Driver API Error " << err_str; \
    }                                                    \
  } while (0);

#define CUDA_RETURN_CALL(status)                         \
  do {                                                   \
    if ((status) != CUDA_SUCCESS) {                      \
      const char *err_str;                               \
      cuGetErrorString((status), &err_str);              \
      LOG(ERROR) << "CUDA Driver API Error " << err_str; \
      return false;                                      \
    }                                                    \
  } while (0);

#define CUDA_RETURN_VAL_IF_FAIL(status, ret)             \
  do {                                                   \
    if ((status) != CUDA_SUCCESS) {                      \
      const char *err_str;                               \
      cuGetErrorString((status), &err_str);              \
      LOG(ERROR) << "CUDA Driver API Error " << err_str; \
      return ret;                                        \
    }                                                    \
  } while (0);

#endif  // TENSORRT_ENGINE_CUDA_MACROS_H
