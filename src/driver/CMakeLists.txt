# src/driver/CMakeLists.txt

file(GLOB DRIVER_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
)
set(DRIVER_SRCS ${DRIVER_SRCS} PARENT_SCOPE)
