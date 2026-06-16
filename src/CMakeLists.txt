# src/CMakeLists.txt

add_subdirectory(common)
add_subdirectory(device)
add_subdirectory(driver)
add_subdirectory(tensorrt_engine)
add_subdirectory(nvtx)
add_subdirectory(plugins)
add_subdirectory(kernels)

set(SRC_SRCS ${COMMON_SRCS} ${DEVICE_SRCS} ${DRIVER_SRCS} ${TENSORRT_ENGINE_SRCS} ${NVTX_SRCS} ${PLUGINS_SRCS} ${KERNELS_SRCS})
set(SRC_SRCS ${SRC_SRCS} PARENT_SCOPE)
