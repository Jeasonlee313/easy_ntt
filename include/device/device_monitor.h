//
// Created by jeason on 2/3/26.
//

#ifndef TENSORRT_ENGINE_DEVICE_MONITOR_H
#define TENSORRT_ENGINE_DEVICE_MONITOR_H

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>

namespace device {
namespace monitor {
bool GetDeviceName(std::string &device_name, int32_t device_id = 0);
bool GetCudaDriverVersion(int32_t &cuda_driver_version);
bool GetCudaRuntimeVersion(int32_t &cuda_runtime_version);

}  // namespace monitor
}  // namespace device

#endif  // TENSORRT_ENGINE_DEVICE_MONITOR_H
