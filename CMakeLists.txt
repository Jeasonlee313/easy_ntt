cmake_minimum_required(VERSION 3.14)

project(easy_ntt LANGUAGES CXX CUDA)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")
endif ()

# Default to C++17
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif ()

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-deprecated-declarations)
    add_compile_options(
            $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:-O3>)
endif ()

# Enable testing by default
option(BUILD_TESTING "Build tests" ON)

find_package(CUDA REQUIRED)
find_package(glog REQUIRED)

# Set TensorRT libraries (aarch64: /usr/.../aarch64-linux-gnu; else: TENSORRT_ROOT tree)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64")
    set(TENSORRT_INCLUDE_DIRS /usr/include/aarch64-linux-gnu)
    set(TENSORRT_LIB_DIR /usr/lib/aarch64-linux-gnu)
    set(TENSORRT_LIBRARIES
            ${TENSORRT_LIB_DIR}/libnvinfer.so
            ${TENSORRT_LIB_DIR}/libnvinfer_plugin.so
            ${TENSORRT_LIB_DIR}/libnvonnxparser.so
    )
    message(STATUS "TensorRT (aarch64): ${TENSORRT_INCLUDE_DIRS}; ${TENSORRT_LIB_DIR}")
else ()
    if (DEFINED ENV{TENSORRT_ROOT})
        set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}")
        message(STATUS "Using TENSORRT_ROOT from environment: ${TENSORRT_ROOT}")
    else ()
        set(TENSORRT_ROOT "/usr/local/tensorrt" CACHE PATH "TensorRT installation root")
        message(STATUS "Using ${TENSORRT_ROOT} as TensorRT Root Dir")
    endif ()
    set(TENSORRT_INCLUDE_DIRS ${TENSORRT_ROOT}/include)
    set(TENSORRT_LIBRARIES
            ${TENSORRT_ROOT}/lib/libnvinfer.so
            ${TENSORRT_ROOT}/lib/libnvinfer_plugin.so
            ${TENSORRT_ROOT}/lib/libnvonnxparser.so
    )
endif ()

# Add include directories
include_directories(
        include
)
include_directories(SYSTEM
        ${TENSORRT_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    include_directories(
            "$ENV{CUDA_HOME}/include/nvtx3"
    )
endif ()

# Find NVTX (optional) — required for nvtx profiling in tensorrt_engine.cpp
find_path(NVTX_INCLUDE_DIR nvToolsExt.h
    PATHS ${CUDA_INCLUDE_DIRS}
          ${CUDA_TOOLKIT_ROOT_DIR}/include
          $ENV{CUDA_HOME}/include
          /usr/local/cuda/include
          /usr/local/cuda/targets/aarch64-linux/include
    NO_DEFAULT_PATH
)
find_library(NVTX_LIBRARY nvToolsExt
    PATHS ${CUDA_LIB_DIR}
          ${CUDA_TOOLKIT_ROOT_DIR}/lib64
          $ENV{CUDA_HOME}/lib64
          /usr/local/cuda/lib64
          /usr/local/cuda/targets/aarch64-linux/lib
          /usr/lib/aarch64-linux-gnu
    NO_DEFAULT_PATH
)

if (NVTX_INCLUDE_DIR AND NVTX_LIBRARY)
    message(STATUS "Found NVTX: ${NVTX_LIBRARY}")
    include_directories(${NVTX_INCLUDE_DIR})
    add_compile_definitions(NVTX_AVAILABLE)
    set(NVTX_LIB ${NVTX_LIBRARY})
else()
    message(STATUS "NVTX not found, NVTX profiling will be disabled")
    set(NVTX_LIB "")
endif()

add_subdirectory(src)

# Create easy_ntt library
add_library(${PROJECT_NAME} SHARED ${SRC_SRCS})

target_include_directories(${PROJECT_NAME} PUBLIC include)

target_link_libraries(${PROJECT_NAME} PRIVATE
        ${CUDA_LIBRARIES}
        ${TENSORRT_LIBRARIES}
        glog::glog
)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>,$<CONFIG:MinSizeRel>>:-fno-semantic-interposition>)
endif ()
if (NVTX_LIB)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${NVTX_LIB})
endif ()

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64")
    set(_MI_CUDA_ARCHS "87" CACHE STRING "CUDA virtual architectures for easy_ntt")
else ()
    set(_MI_CUDA_ARCHS "75;80;86;89" CACHE STRING "CUDA virtual architectures for easy_ntt")
endif ()
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "${_MI_CUDA_ARCHS}")
#set_target_properties(${PROJECT_NAME} PROPERTIES VERSION "0.1.0" SOVERSION "0")


if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    string(APPEND CMAKE_CUDA_FLAGS_RELEASE " -O3")
    string(APPEND CMAKE_CUDA_FLAGS_RELWITHDEBINFO " -O3")
    string(APPEND CMAKE_CUDA_FLAGS_MINSIZEREL " -O3")
endif ()

option(EASY_NTT_ENABLE_IPO "Release/RelWithDebInfo 下启用 LTO" OFF)
include(CheckIPOSupported)
check_ipo_supported(RESULT _mi_ipo_ok OUTPUT _mi_ipo_err)
if (EASY_NTT_ENABLE_IPO AND _mi_ipo_ok)
    set_property(TARGET ${PROJECT_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set_property(TARGET ${PROJECT_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
elseif (EASY_NTT_ENABLE_IPO)
    message(STATUS "IPO/LTO not supported for easy_ntt: ${_mi_ipo_err}")
endif ()
option(EASY_NTT_ENABLE_BSYMBOLIC "GNU 链接使用 -Wl,-Bsymbolic-functions（独立 Makefile 未使用）" OFF)
if (EASY_NTT_ENABLE_BSYMBOLIC AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_options(${PROJECT_NAME} PRIVATE
            $<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:-Wl,-Bsymbolic-functions>)
endif ()

# Install libraries
install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

# Install header files
install(DIRECTORY include/
        DESTINATION include
        FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h" PATTERN "*.cuh")

# Install documentation
install(DIRECTORY doc/
        DESTINATION share/doc/easy_ntt
        FILES_MATCHING PATTERN "*.md")

# Testing
if (BUILD_TESTING)
    find_package(GTest REQUIRED)
    enable_testing()
    include(GoogleTest)
    add_subdirectory(test)
endif ()
