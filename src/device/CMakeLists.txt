# src/device/CMakeLists.txt

file(GLOB DEVICE_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
)
set(DEVICE_SRCS ${DEVICE_SRCS} PARENT_SCOPE)
