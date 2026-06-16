//
// Created by jeason on 2/3/26.
//

#include "../../include/device/device_monitor.h"

#include <memory>

#include <gtest/gtest.h>

#include "driver/cuda_macros.h"

namespace device {
namespace monitor {

TEST(DeviceNameTest, DeviceNameTest) {
  std::string device_name = "";
  EXPECT_TRUE(GetDeviceName(device_name));
  CUINFO << "grab device name: " << device_name;
}

TEST(DeviceVersionTest, DeviceVersionTest) {
  int32_t cuda_driver_version = -1;
  EXPECT_TRUE(GetCudaDriverVersion(cuda_driver_version));
  CUINFO << "grab cuda driver version: " << cuda_driver_version;
  int32_t cuda_runtime_version = -1;
  EXPECT_TRUE(GetCudaRuntimeVersion(cuda_runtime_version));
  CUINFO << "grab cuda runtime version: " << cuda_runtime_version;
}

}  // namespace monitor
}  // namespace device
