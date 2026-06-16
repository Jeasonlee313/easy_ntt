//
// Created by jeason on 2/3/26.
//

#include "device/device_monitor.h"

#include "driver/cuda_macros.h"

namespace device {
namespace monitor {

bool GetDeviceName(std::string &device_name, int32_t device_id) {
  cudaDeviceProp prop{};
  CUDART_RETURN_VAL_IF_FAIL(cudaGetDeviceProperties(&prop, device_id), false);
  device_name = std::string(prop.name);
  size_t pos = 0;
  while ((pos = device_name.find(" ", pos)) != std::string::npos) {
    device_name.replace(pos, 1, "_");
    ++pos;
  }
  return true;
}

bool GetCudaDriverVersion(int32_t &driver_version) {
  CUDART_RETURN_VAL_IF_FAIL(cudaDriverGetVersion(&driver_version), false);
  return true;
}

bool GetCudaRuntimeVersion(int32_t &runtime_version) {
  CUDART_RETURN_VAL_IF_FAIL(cudaRuntimeGetVersion(&runtime_version), false);
  return true;
}

}  // namespace monitor
}  // namespace device
