# test/tensorrt_engine/CMakeLists.txt

find_package(OpenCV REQUIRED)

file(GLOB TENSORRT_ENGINE_TEST_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

# Use standard GTest
add_executable(tensorrt_engine_test ${TENSORRT_ENGINE_TEST_SRCS})
target_include_directories(
        tensorrt_engine_test
        PUBLIC
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
        tensorrt_engine_test
        PRIVATE
        GTest::GTest
        GTest::Main
        ${CUDA_LIBRARIES}
        glog::glog
        ${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${TENSORRT_LIBRARIES}
        curl
)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test_data
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
gtest_discover_tests(tensorrt_engine_test)
