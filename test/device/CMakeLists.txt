# test/device/CMakeLists.txt

file(GLOB DEVICE_TEST_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

add_executable(device_test ${DEVICE_TEST_SRCS})
target_link_libraries(device_test GTest::GTest GTest::Main)

target_link_libraries(device_test ${CUDA_LIBRARIES} glog ${PROJECT_NAME})
gtest_discover_tests(device_test)
