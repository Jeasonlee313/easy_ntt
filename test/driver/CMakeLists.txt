# test/driver/CMakeLists.txt

file(GLOB DRIVER_TEST_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

# Use standard GTest
add_executable(driver_test ${DRIVER_TEST_SRCS})
target_link_libraries(driver_test GTest::GTest GTest::Main)

target_link_libraries(driver_test ${CUDA_LIBRARIES} glog ${PROJECT_NAME})
gtest_discover_tests(driver_test)
