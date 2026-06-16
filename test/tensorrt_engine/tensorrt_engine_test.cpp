//
// Created by jeason on 2/6/26.
//
#include "tensorrt_engine/tensorrt_engine.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <thread>

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include "driver/cuda_macros.h"

namespace tensorrt_engine {
namespace fs = std::filesystem;
constexpr uint32_t kSleep = 5;

Logger kTestLogger;

class ModelInferenceTest : public ::testing::Test {
 public:
  void SetUp() override { std::this_thread::sleep_for(std::chrono::milliseconds(kSleep)); }

  void TearDown() override { std::this_thread::sleep_for(std::chrono::milliseconds(kSleep)); }

 protected:
  static std::shared_ptr<TensorRTEngine> LoadEngine() {
    const std::string kEnginePath = "test_data/resnet/ResNet50.engine";

    // CI/CD environments typically don't have the prebuilt engine file.
    // Skip the test instead of trying to build from ONNX (which requires
    // full TensorRT builder libraries and is too slow for CI).
    if (!fs::exists(kEnginePath)) {
      return nullptr;
    }

    auto model = std::make_shared<TensorRTEngine>(kEnginePath);
    return model;
  }

  static void PreprocessInput(float *input_data) {
    const std::string kImage = "test_data/resnet/binoculars.jpeg";
    constexpr uint32_t kInputH = 224;
    constexpr uint32_t kInputW = 224;

    cv::Mat image = cv::imread(kImage);
    if (image.empty()) {
      return;
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    cv::resize(image, image, cv::Size(kInputW, kInputH));
    image.convertTo(image, CV_32FC3, 1.0 / 255);

    for (uint32_t h = 0; h < kInputH; ++h) {
      for (uint32_t w = 0; w < kInputW; ++w) {
        input_data[0 * kInputH * kInputW + h * kInputW + w] = (image.at<cv::Vec3f>(h, w)[0] - 0.485f) / 0.229f;
        input_data[1 * kInputH * kInputW + h * kInputW + w] = (image.at<cv::Vec3f>(h, w)[1] - 0.456f) / 0.224f;
        input_data[2 * kInputH * kInputW + h * kInputW + w] = (image.at<cv::Vec3f>(h, w)[2] - 0.406f) / 0.225f;
      }
    }
  }
};

TEST_F(ModelInferenceTest, ResNet50Test) {
  const std::string kInputName = "data";
  const std::string kOutputName = "resnetv17_dense0_fwd";
  constexpr uint32_t kInputH = 224;
  constexpr uint32_t kInputW = 224;
  constexpr uint32_t kOutputLen = 1000;

  auto model = LoadEngine();
  if (!model || !model->IsLoaded()) {
    GTEST_SKIP() << "Engine not available, skip ResNet50Test";
  }

  float input_data[3 * kInputW * kInputH];
  PreprocessInput(input_data);

  float output_data[kOutputLen];

  // Dynamic engine: must set binding shape before SetInputData
  EXPECT_TRUE(model->SetBindingShape(kInputName, nvinfer1::Dims4{1, 3, kInputH, kInputW}));
  EXPECT_TRUE(model->SetInputData(kInputName, input_data, 3 * kInputH * kInputW * sizeof(float),
                                  {1, 3, kInputH, kInputW}, false));
  EXPECT_TRUE(model->Infer());
  EXPECT_TRUE(model->GetOutputData(kOutputName, output_data, kOutputLen * sizeof(float), false));

  // Find the index with the highest score
  int max_index = 0;
  float max_value = output_data[0];
  for (uint32_t i = 1; i < kOutputLen; ++i) {
    if (output_data[i] > max_value) {
      max_value = output_data[i];
      max_index = i;
    }
  }
  CUINFO << "max_index: " << max_index << " max_value: " << max_value;
  EXPECT_EQ(max_index, 447);  // Expected class index for the input image
}

TEST_F(ModelInferenceTest, ResNet50AsyncTest) {
  const std::string kInputName = "data";
  const std::string kOutputName = "resnetv17_dense0_fwd";
  constexpr uint32_t kInputH = 224;
  constexpr uint32_t kInputW = 224;
  constexpr uint32_t kOutputLen = 1000;

  auto model = LoadEngine();
  if (!model || !model->IsLoaded()) {
    GTEST_SKIP() << "Engine not available, skip ResNet50AsyncTest";
  }

  float input_data[3 * kInputW * kInputH];
  PreprocessInput(input_data);

  float output_data[kOutputLen];

  // Dynamic engine: must set binding shape before SetInputData
  EXPECT_TRUE(model->SetBindingShape(kInputName, nvinfer1::Dims4{1, 3, kInputH, kInputW}));

  // Async H2D + InferAsync + sync D2H (sync D2H will implicitly wait for stream)
  EXPECT_TRUE(model->SetInputData(kInputName, input_data, 3 * kInputH * kInputW * sizeof(float),
                                  {1, 3, kInputH, kInputW}, true));
  EXPECT_TRUE(model->InferAsync());
  // Sync before reading output to ensure enqueueV3 has finished writing.
  EXPECT_TRUE(model->Synchronize());
  EXPECT_TRUE(model->GetOutputData(kOutputName, output_data, kOutputLen * sizeof(float), false));

  int max_index = 0;
  float max_value = output_data[0];
  for (uint32_t i = 1; i < kOutputLen; ++i) {
    if (output_data[i] > max_value) {
      max_value = output_data[i];
      max_index = i;
    }
  }
  CUINFO << "async max_index: " << max_index << " max_value: " << max_value;
  EXPECT_EQ(max_index, 447);  // Same expected result as sync path
}

TEST_F(ModelInferenceTest, BindingQueryTest) {
  const std::string kInputName = "data";
  const std::string kOutputName = "resnetv17_dense0_fwd";
  constexpr uint32_t kInputH = 224;
  constexpr uint32_t kInputW = 224;
  constexpr uint32_t kMaxBatch = 4;
  constexpr uint32_t kOutputLen = 1000;

  auto model = LoadEngine();
  if (!model || !model->IsLoaded()) {
    GTEST_SKIP() << "Engine not available, skip BindingQueryTest";
  }

  EXPECT_TRUE(model->IsLoaded());

  // GetBindingByName
  auto input_tensor = model->GetBindingByName(kInputName);
  ASSERT_NE(input_tensor, nullptr);
  auto output_tensor = model->GetBindingByName(kOutputName);
  ASSERT_NE(output_tensor, nullptr);
  EXPECT_EQ(model->GetBindingByName("nonexistent"), nullptr);

  // Tensor metadata (dynamic engine: allocated at max profile shape)
  EXPECT_EQ(input_tensor->name(), kInputName);
  EXPECT_TRUE(input_tensor->is_input());
  EXPECT_FALSE(input_tensor->is_output());
  EXPECT_EQ(input_tensor->dataType(), nvinfer1::DataType::kFLOAT);
  EXPECT_EQ(input_tensor->element_size(), 4);
  EXPECT_EQ(input_tensor->byte_size(), kMaxBatch * 3 * kInputH * kInputW * sizeof(float));
  EXPECT_EQ(input_tensor->num_elements(), kMaxBatch * 3 * kInputH * kInputW);

  const auto &input_dims = input_tensor->dims();
  EXPECT_EQ(input_dims.nbDims, 4);
  EXPECT_EQ(input_dims.d[0], static_cast<int>(kMaxBatch));
  EXPECT_EQ(input_dims.d[1], 3);
  EXPECT_EQ(input_dims.d[2], static_cast<int>(kInputH));
  EXPECT_EQ(input_dims.d[3], static_cast<int>(kInputW));

  // Output is now pre-allocated at max profile shape after construction
  EXPECT_EQ(output_tensor->name(), kOutputName);
  EXPECT_TRUE(output_tensor->is_output());
  EXPECT_FALSE(output_tensor->is_input());
  EXPECT_EQ(output_tensor->dataType(), nvinfer1::DataType::kFLOAT);
  EXPECT_EQ(output_tensor->byte_size(), kMaxBatch * kOutputLen * sizeof(float));
  EXPECT_EQ(output_tensor->num_elements(), kMaxBatch * kOutputLen);

  const auto &output_dims = output_tensor->dims();
  EXPECT_EQ(output_dims.nbDims, 2);
  EXPECT_EQ(output_dims.d[0], static_cast<int>(kMaxBatch));
  EXPECT_EQ(output_dims.d[1], static_cast<int>(kOutputLen));

  // GetTensorMap
  const auto &tensor_map = model->GetTensorMap();
  EXPECT_EQ(tensor_map.size(), 2);
  EXPECT_NE(tensor_map.find(kInputName), tensor_map.end());
  EXPECT_NE(tensor_map.find(kOutputName), tensor_map.end());
}

TEST_F(ModelInferenceTest, SizeMismatchTest) {
  const std::string kInputName = "data";
  const std::string kOutputName = "resnetv17_dense0_fwd";
  constexpr uint32_t kInputH = 224;
  constexpr uint32_t kInputW = 224;
  constexpr uint32_t kOutputLen = 1000;

  auto model = LoadEngine();
  if (!model || !model->IsLoaded()) {
    GTEST_SKIP() << "Engine not available, skip SizeMismatchTest";
  }

  float input_data[3 * kInputW * kInputH];
  float output_data[kOutputLen];

  // Dynamic engine: must set binding shape first so byte_size matches
  EXPECT_TRUE(model->SetBindingShape(kInputName, nvinfer1::Dims4{1, 3, kInputH, kInputW}));

  size_t correct_input_size = 3 * kInputH * kInputW * sizeof(float);
  size_t correct_output_size = kOutputLen * sizeof(float);

  // Wrong input size
  EXPECT_FALSE(model->SetInputData(kInputName, input_data, correct_input_size - 1, {1, 3, kInputH, kInputW}, false));
  EXPECT_FALSE(model->SetInputData(kInputName, input_data, correct_input_size + 1, {1, 3, kInputH, kInputW}, false));

  // Wrong output size
  EXPECT_FALSE(model->GetOutputData(kOutputName, output_data, correct_output_size - 1, false));
  EXPECT_FALSE(model->GetOutputData(kOutputName, output_data, correct_output_size + 1, false));

  // Wrong input dims
  EXPECT_FALSE(model->SetInputData(kInputName, input_data, correct_input_size, {1, 3, kInputH, kInputW - 1}, false));
  EXPECT_FALSE(model->SetInputData(kInputName, input_data, correct_input_size, {1, 3, kInputH}, false));

  // Correct size should still work
  EXPECT_TRUE(model->SetInputData(kInputName, input_data, correct_input_size, {1, 3, kInputH, kInputW}, false));
}

TEST_F(ModelInferenceTest, InvalidNameTest) {
  const std::string kInputName = "data";
  const std::string kOutputName = "resnetv17_dense0_fwd";
  constexpr uint32_t kInputH = 224;
  constexpr uint32_t kInputW = 224;
  constexpr uint32_t kOutputLen = 1000;

  auto model = LoadEngine();
  if (!model || !model->IsLoaded()) {
    GTEST_SKIP() << "Engine not available, skip InvalidNameTest";
  }

  float input_data[3 * kInputW * kInputH];
  float output_data[kOutputLen];

  // Nonexistent name
  EXPECT_FALSE(model->SetInputData("nonexistent", input_data, sizeof(input_data), {}, false));
  EXPECT_FALSE(model->GetOutputData("nonexistent", output_data, sizeof(output_data), false));

  // Wrong direction: set input on output tensor
  EXPECT_FALSE(model->SetInputData(kOutputName, output_data, sizeof(output_data), {}, false));

  // Wrong direction: get output from input tensor
  EXPECT_FALSE(model->GetOutputData(kInputName, input_data, sizeof(input_data), false));
}

TEST_F(ModelInferenceTest, FreezeIOStaticTest) {
  const std::string kInputName = "data";
  const std::string kOutputName = "resnetv17_dense0_fwd";
  constexpr uint32_t kInputH = 224;
  constexpr uint32_t kInputW = 224;
  constexpr uint32_t kOutputLen = 1000;

  auto model = LoadEngine();
  if (!model || !model->IsLoaded()) {
    GTEST_SKIP() << "Engine not available, skip FreezeIOStaticTest";
  }

  // Dynamic engine: set shape, run once to resolve outputs, then freeze
  EXPECT_TRUE(model->SetBindingShape(kInputName, nvinfer1::Dims4{1, 3, kInputH, kInputW}));

  float input_data[3 * kInputW * kInputH];
  float output_data[kOutputLen];
  PreprocessInput(input_data);

  EXPECT_TRUE(model->SetInputData(kInputName, input_data, 3 * kInputH * kInputW * sizeof(float),
                                  {1, 3, kInputH, kInputW}, false));
  EXPECT_TRUE(model->Infer());

  EXPECT_FALSE(model->IsIOFrozen());
  EXPECT_TRUE(model->FreezeIO());
  EXPECT_TRUE(model->IsIOFrozen());

  // Set input data (frozen state still allows data copy)
  EXPECT_TRUE(model->SetInputData(kInputName, input_data, 3 * kInputH * kInputW * sizeof(float),
                                  {1, 3, kInputH, kInputW}, false));

  // Run inference twice via fast path
  EXPECT_TRUE(model->InferAsync());
  EXPECT_TRUE(model->InferAsync());

  // Sync and fetch output
  EXPECT_TRUE(model->GetOutputData(kOutputName, output_data, kOutputLen * sizeof(float), false));

  int max_index = 0;
  float max_value = output_data[0];
  for (uint32_t i = 1; i < kOutputLen; ++i) {
    if (output_data[i] > max_value) {
      max_value = output_data[i];
      max_index = i;
    }
  }
  EXPECT_EQ(max_index, 447);
}

TEST_F(ModelInferenceTest, ExecuteStaticTest) {
  const std::string kInputName = "data";
  const std::string kOutputName = "resnetv17_dense0_fwd";
  constexpr uint32_t kInputH = 224;
  constexpr uint32_t kInputW = 224;
  constexpr uint32_t kOutputLen = 1000;

  auto model = LoadEngine();
  if (!model || !model->IsLoaded()) {
    GTEST_SKIP() << "Engine not available, skip ExecuteStaticTest";
  }

  // Dynamic engine: must set binding shape before SetInputData
  EXPECT_TRUE(model->SetBindingShape(kInputName, nvinfer1::Dims4{1, 3, kInputH, kInputW}));

  float input_data[3 * kInputW * kInputH];
  float output_data[kOutputLen];
  PreprocessInput(input_data);

  // Set input via legacy API, then call no-arg Execute
  EXPECT_TRUE(model->SetInputData(kInputName, input_data, 3 * kInputH * kInputW * sizeof(float),
                                  {1, 3, kInputH, kInputW}, false));
  EXPECT_TRUE(model->Execute());
  EXPECT_TRUE(model->GetOutputData(kOutputName, output_data, kOutputLen * sizeof(float), false));

  int max_index = 0;
  float max_value = output_data[0];
  for (uint32_t i = 1; i < kOutputLen; ++i) {
    if (output_data[i] > max_value) {
      max_value = output_data[i];
      max_index = i;
    }
  }
  EXPECT_EQ(max_index, 447);
}

TEST_F(ModelInferenceTest, ExecuteWithPointersStaticTest) {
  const std::string kInputName = "data";
  const std::string kOutputName = "resnetv17_dense0_fwd";
  constexpr uint32_t kInputH = 224;
  constexpr uint32_t kInputW = 224;
  constexpr uint32_t kOutputLen = 1000;

  auto model = LoadEngine();
  if (!model || !model->IsLoaded()) {
    GTEST_SKIP() << "Engine not available, skip ExecuteWithPointersStaticTest";
  }

  // Dynamic engine: must set binding shape before Execute
  EXPECT_TRUE(model->SetBindingShape(kInputName, nvinfer1::Dims4{1, 3, kInputH, kInputW}));

  float input_data[3 * kInputW * kInputH];
  float output_data[kOutputLen];
  PreprocessInput(input_data);

  std::unordered_map<std::string, void *> inputs;
  std::unordered_map<std::string, void *> outputs;
  inputs[kInputName] = input_data;
  outputs[kOutputName] = output_data;

  EXPECT_TRUE(model->Execute(inputs, outputs));

  int max_index = 0;
  float max_value = output_data[0];
  for (uint32_t i = 1; i < kOutputLen; ++i) {
    if (output_data[i] > max_value) {
      max_value = output_data[i];
      max_index = i;
    }
  }
  EXPECT_EQ(max_index, 447);
}

TEST_F(ModelInferenceTest, DynamicShapeFreezeIORejectsOutOfBoundsShapeTest) {
  const std::string kInputName = "data";
  constexpr uint32_t kInputH = 224;
  constexpr uint32_t kInputW = 224;
  constexpr uint32_t kMaxBatch = 4;

  auto model = LoadEngine();
  if (!model || !model->IsLoaded()) {
    GTEST_SKIP() << "Engine not available, skip DynamicShapeFreezeIORejectsOutOfBoundsShapeTest";
  }

  // Dynamic engine: set shape to max, run once to resolve outputs, then freeze
  EXPECT_TRUE(model->SetBindingShape(
      kInputName, nvinfer1::Dims4{static_cast<int64_t>(kMaxBatch), 3, kInputH, kInputW}));
  std::vector<float> input_data(kMaxBatch * 3 * kInputH * kInputW, 1.0f);
  EXPECT_TRUE(model->SetInputData(kInputName, input_data.data(), input_data.size() * sizeof(float),
                                  {static_cast<int64_t>(kMaxBatch), 3, kInputH, kInputW}, false));
  EXPECT_TRUE(model->Infer());

  EXPECT_TRUE(model->FreezeIO());
  EXPECT_TRUE(model->IsIOFrozen());

  // Shape beyond max profile should still be rejected after freezing
  EXPECT_FALSE(model->SetBindingShape(
      kInputName, nvinfer1::Dims4{static_cast<int64_t>(kMaxBatch) + 1, 3, kInputH, kInputW}));
}

TEST(ModelInferenceStandaloneTest, FreezeIONotLoadedTest) {
  TensorRTEngine engine("nonexistent.engine");
  EXPECT_FALSE(engine.IsLoaded());
  EXPECT_FALSE(engine.FreezeIO());
  EXPECT_FALSE(engine.IsIOFrozen());
  EXPECT_FALSE(engine.Execute());

  std::unordered_map<std::string, void *> inputs, outputs;
  EXPECT_FALSE(engine.Execute(inputs, outputs));
}

TEST_F(ModelInferenceTest, DynamicShapeBasicTest) {
  const std::string kInputName = "data";
  const std::string kOutputName = "resnetv17_dense0_fwd";
  constexpr uint32_t kC = 3;
  constexpr uint32_t kH = 224;
  constexpr uint32_t kW = 224;
  constexpr uint32_t kOutputLen = 1000;

  auto model = LoadEngine();
  if (!model || !model->IsLoaded()) {
    GTEST_SKIP() << "Engine not available, skip DynamicShapeBasicTest";
  }

  // Verify dynamic binding detection
  EXPECT_TRUE(model->IsDynamicBinding(kInputName));
  EXPECT_TRUE(model->IsDynamicBinding(kOutputName));

  // Helper lambda to run inference with a given batch size
  auto infer_with_batch = [&](int batch) -> bool {
    nvinfer1::Dims4 input_dims{batch, kC, kH, kW};
    if (!model->SetBindingShape(kInputName, input_dims)) return false;

    auto input_tensor = model->GetBindingByName(kInputName);
    if (!input_tensor->Reshape(input_dims)) return false;

    std::vector<float> input_data(batch * kC * kH * kW, 1.0f);
    if (!input_tensor->CopyFromHostAsync(input_data.data(), input_tensor->stream())) return false;
    if (!model->InferAsync()) return false;
    auto output_shape = model->GetTensorShape(kOutputName);
    return output_shape.nbDims == 2 && output_shape.d[0] == batch && output_shape.d[1] == kOutputLen;
  };

  // Test batch = 1
  EXPECT_TRUE(infer_with_batch(1));
  auto out_shape = model->GetTensorShape(kOutputName);
  EXPECT_EQ(out_shape.nbDims, 2);
  EXPECT_EQ(out_shape.d[0], 1);
  EXPECT_EQ(out_shape.d[1], kOutputLen);

  // Test batch = 2
  EXPECT_TRUE(infer_with_batch(2));
  out_shape = model->GetTensorShape(kOutputName);
  EXPECT_EQ(out_shape.d[0], 2);

  // Test batch = 4 (max profile)
  EXPECT_TRUE(infer_with_batch(4));
  out_shape = model->GetTensorShape(kOutputName);
  EXPECT_EQ(out_shape.d[0], 4);
}

TEST_F(ModelInferenceTest, DynamicShapeFreezeIOTest) {
  const std::string kInputName = "data";
  const std::string kOutputName = "resnetv17_dense0_fwd";
  constexpr uint32_t kInputH = 224;
  constexpr uint32_t kInputW = 224;
  constexpr uint32_t kOutputLen = 1000;
  constexpr uint32_t kMaxBatch = 4;

  auto model = LoadEngine();
  if (!model || !model->IsLoaded()) {
    GTEST_SKIP() << "Engine not available, skip DynamicShapeFreezeIOTest";
  }

  EXPECT_TRUE(model->IsDynamicBinding(kInputName));

  auto input_tensor = model->GetBindingByName(kInputName);
  ASSERT_NE(input_tensor, nullptr);

  // Set batch=2 and run one inference to fully resolve output shapes
  EXPECT_TRUE(model->SetBindingShape(kInputName, nvinfer1::Dims4{2, 3, kInputH, kInputW}));
  EXPECT_TRUE(input_tensor->Reshape(nvinfer1::Dims4{2, 3, kInputH, kInputW}));
  std::vector<float> input_data(2 * 3 * kInputH * kInputW, 1.0f);
  EXPECT_TRUE(input_tensor->CopyFromHostAsync(input_data.data(), input_tensor->stream()));
  EXPECT_TRUE(model->Infer());

  // Freeze I/O (dynamic engine: addresses locked, shapes may still vary within max profile)
  EXPECT_TRUE(model->FreezeIO());
  EXPECT_TRUE(model->IsIOFrozen());

  // Run inference twice via fast path, then sync with infer()
  EXPECT_TRUE(model->InferAsync());
  EXPECT_TRUE(model->InferAsync());
  EXPECT_TRUE(model->Infer());

  // Change shape within max profile should succeed after freezing
  EXPECT_TRUE(model->SetBindingShape(
      kInputName, nvinfer1::Dims4{static_cast<int64_t>(kMaxBatch), 3, kInputH, kInputW}));
  EXPECT_TRUE(input_tensor->Reshape(
      nvinfer1::Dims4{static_cast<int64_t>(kMaxBatch), 3, kInputH, kInputW}));
  std::vector<float> input_data_max(kMaxBatch * 3 * kInputH * kInputW, 1.0f);
  EXPECT_TRUE(input_tensor->CopyFromHostAsync(input_data_max.data(), input_tensor->stream()));
  EXPECT_TRUE(model->Infer());

  std::vector<float> output_data_max(kMaxBatch * kOutputLen);
  EXPECT_TRUE(model->GetOutputData(kOutputName, output_data_max.data(), output_data_max.size() * sizeof(float), false));
  auto out_shape = model->GetTensorShape(kOutputName);
  EXPECT_EQ(out_shape.nbDims, 2);
  EXPECT_EQ(out_shape.d[0], static_cast<int>(kMaxBatch));
  EXPECT_EQ(out_shape.d[1], static_cast<int>(kOutputLen));

  // Shape beyond max profile should still be rejected
  EXPECT_FALSE(model->SetBindingShape(
      kInputName, nvinfer1::Dims4{static_cast<int64_t>(kMaxBatch) + 1, 3, kInputH, kInputW}));
}

TEST_F(ModelInferenceTest, DynamicShapeExecuteTest) {
  const std::string kInputName = "data";
  const std::string kOutputName = "resnetv17_dense0_fwd";
  constexpr uint32_t kC = 3;
  constexpr uint32_t kH = 224;
  constexpr uint32_t kW = 224;
  constexpr uint32_t kOutputLen = 1000;

  auto model = LoadEngine();
  if (!model || !model->IsLoaded()) {
    GTEST_SKIP() << "Engine not available, skip DynamicShapeExecuteTest";
  }

  // Set shape via SetBindingShape, then use Execute()
  EXPECT_TRUE(model->SetBindingShape(kInputName, nvinfer1::Dims4{2, kC, kH, kW}));

  std::vector<float> input_data(2 * kC * kH * kW, 1.0f);
  std::vector<float> output_data(2 * kOutputLen);

  std::unordered_map<std::string, void *> inputs;
  std::unordered_map<std::string, void *> outputs;
  inputs[kInputName] = input_data.data();
  outputs[kOutputName] = output_data.data();

  EXPECT_TRUE(model->Execute(inputs, outputs));

  // Verify output shape updated
  auto out_shape = model->GetTensorShape(kOutputName);
  EXPECT_EQ(out_shape.nbDims, 2);
  EXPECT_EQ(out_shape.d[0], 2);
  EXPECT_EQ(out_shape.d[1], kOutputLen);
}

TEST_F(ModelInferenceTest, DynamicShapeFreezeIOAddressLockedTest) {
  const std::string kInputName = "data";
  const std::string kOutputName = "resnetv17_dense0_fwd";
  constexpr uint32_t kInputH = 224;
  constexpr uint32_t kInputW = 224;

  auto model = LoadEngine();
  if (!model || !model->IsLoaded()) {
    GTEST_SKIP() << "Engine not available, skip DynamicShapeFreezeIOAddressLockedTest";
  }

  EXPECT_TRUE(model->SetBindingShape(kInputName, nvinfer1::Dims4{1, 3, kInputH, kInputW}));
  std::vector<float> input_data(1 * 3 * kInputH * kInputW, 1.0f);
  EXPECT_TRUE(model->SetInputData(kInputName, input_data.data(), input_data.size() * sizeof(float),
                                  {1, 3, kInputH, kInputW}, false));
  EXPECT_TRUE(model->Infer());

  EXPECT_TRUE(model->FreezeIO());

  auto input_tensor = model->GetBindingByName(kInputName);
  auto output_tensor = model->GetBindingByName(kOutputName);
  ASSERT_NE(input_tensor, nullptr);
  ASSERT_NE(output_tensor, nullptr);

  // Changing binding address should be rejected while frozen.
  EXPECT_FALSE(model->SetBindingAddress(kInputName, const_cast<void *>(input_tensor->data())));
  EXPECT_FALSE(model->SetBindingAddress(kOutputName, const_cast<void *>(output_tensor->data())));

  // But changing shape within max profile is still allowed.
  EXPECT_TRUE(model->SetBindingShape(kInputName, nvinfer1::Dims4{2, 3, kInputH, kInputW}));
}

TEST_F(ModelInferenceTest, FreezeIOOutputShapeUpdatesTest) {
  const std::string kInputName = "data";
  const std::string kOutputName = "resnetv17_dense0_fwd";
  constexpr uint32_t kInputH = 224;
  constexpr uint32_t kInputW = 224;
  constexpr uint32_t kOutputLen = 1000;
  constexpr uint32_t kMaxBatch = 4;

  auto model = LoadEngine();
  if (!model || !model->IsLoaded()) {
    GTEST_SKIP() << "Engine not available, skip FreezeIOOutputShapeUpdatesTest";
  }

  auto input_tensor = model->GetBindingByName(kInputName);
  ASSERT_NE(input_tensor, nullptr);

  auto run_with_batch = [&](int batch) -> bool {
    if (!model->SetBindingShape(kInputName, nvinfer1::Dims4{batch, 3, kInputH, kInputW})) return false;
    if (!input_tensor->Reshape(nvinfer1::Dims4{batch, 3, kInputH, kInputW})) return false;
    std::vector<float> input_data(batch * 3 * kInputH * kInputW, 1.0f);
    if (!input_tensor->CopyFromHostAsync(input_data.data(), input_tensor->stream())) return false;
    if (!model->Infer()) return false;
    std::vector<float> output_data(batch * kOutputLen);
    if (!model->GetOutputData(kOutputName, output_data.data(), output_data.size() * sizeof(float), false)) return false;
    auto shape = model->GetTensorShape(kOutputName);
    return shape.nbDims == 2 && shape.d[0] == batch && shape.d[1] == kOutputLen;
  };

  EXPECT_TRUE(run_with_batch(1));
  EXPECT_TRUE(model->FreezeIO());
  EXPECT_TRUE(model->IsIOFrozen());

  // After freezing, output shape should still update correctly for each new batch.
  EXPECT_TRUE(run_with_batch(2));
  EXPECT_TRUE(run_with_batch(static_cast<int>(kMaxBatch)));
}

TEST_F(ModelInferenceTest, BindingNameQueryTest) {
  const std::string kInputName = "data";
  const std::string kOutputName = "resnetv17_dense0_fwd";

  auto model = LoadEngine();
  if (!model || !model->IsLoaded()) {
    GTEST_SKIP() << "Engine not available, skip BindingNameQueryTest";
  }

  auto input_names = model->GetInputNames();
  auto output_names = model->GetOutputNames();

  EXPECT_EQ(input_names.size(), 1);
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(input_names[0], kInputName);
  EXPECT_EQ(output_names[0], kOutputName);
}

TEST_F(ModelInferenceTest, HasBindingTest) {
  auto model = LoadEngine();
  if (!model || !model->IsLoaded()) {
    GTEST_SKIP() << "Engine not available, skip HasBindingTest";
  }

  EXPECT_TRUE(model->HasInputBinding("data"));
  EXPECT_FALSE(model->HasInputBinding("resnetv17_dense0_fwd"));
  EXPECT_FALSE(model->HasInputBinding("nonexistent"));

  EXPECT_TRUE(model->HasOutputBinding("resnetv17_dense0_fwd"));
  EXPECT_FALSE(model->HasOutputBinding("data"));
  EXPECT_FALSE(model->HasOutputBinding("nonexistent"));
}

TEST_F(ModelInferenceTest, EngineTensorShapeTest) {
  auto model = LoadEngine();
  if (!model || !model->IsLoaded()) {
    GTEST_SKIP() << "Engine not available, skip EngineTensorShapeTest";
  }

  constexpr uint32_t kMaxBatch = 4;

  // Dynamic engine: engine shape has -1 for dynamic batch
  auto engine_shape = model->GetEngineTensorShape("data");
  auto context_shape = model->GetTensorShape("data");
  EXPECT_EQ(engine_shape.nbDims, 4);
  EXPECT_EQ(engine_shape.d[0], -1);  // dynamic batch
  EXPECT_EQ(engine_shape.d[1], 3);
  EXPECT_EQ(engine_shape.d[2], 224);
  EXPECT_EQ(engine_shape.d[3], 224);

  // After construction the context shape reflects the max profile shape
  EXPECT_EQ(context_shape.d[0], static_cast<int>(kMaxBatch));

  // After setting shape, context shape resolves but engine shape stays -1
  EXPECT_TRUE(model->SetBindingShape("data", nvinfer1::Dims4{1, 3, 224, 224}));
  auto resolved_shape = model->GetTensorShape("data");
  EXPECT_EQ(resolved_shape.d[0], 1);

  auto still_dynamic_engine_shape = model->GetEngineTensorShape("data");
  EXPECT_EQ(still_dynamic_engine_shape.d[0], -1);
}

TEST_F(ModelInferenceTest, StreamControlTest) {
  auto model = LoadEngine();
  if (!model || !model->IsLoaded()) {
    GTEST_SKIP() << "Engine not available, skip StreamControlTest";
  }

  // GetCudaStream should return a valid handle
  EXPECT_NE(model->GetCudaStream(), nullptr);

  // Synchronize on a clean stream should succeed
  EXPECT_TRUE(model->Synchronize());

  // SetStream with a raw cudaStream_t (best-effort API)
  cudaStream_t raw_stream = nullptr;
  CUDART_CALL(cudaStreamCreate(&raw_stream));
  EXPECT_TRUE(model->SetStream(raw_stream));
  EXPECT_EQ(model->GetCudaStream(), raw_stream);
  EXPECT_TRUE(model->Synchronize());
  CUDART_CALL(cudaStreamDestroy(raw_stream));
}

TEST_F(ModelInferenceTest, DynamicEngineTensorShapeTest) {
  auto model = LoadEngine();
  if (!model || !model->IsLoaded()) {
    GTEST_SKIP() << "Engine not available, skip DynamicEngineTensorShapeTest";
  }

  constexpr uint32_t kMaxBatch = 4;

  // Engine-level shape should contain -1 for dynamic batch
  auto engine_input_shape = model->GetEngineTensorShape("data");
  EXPECT_EQ(engine_input_shape.nbDims, 4);
  EXPECT_EQ(engine_input_shape.d[0], -1);  // dynamic batch
  EXPECT_EQ(engine_input_shape.d[1], 3);
  EXPECT_EQ(engine_input_shape.d[2], 224);
  EXPECT_EQ(engine_input_shape.d[3], 224);

  // After construction the context shape reflects the max profile shape
  auto context_input_shape = model->GetTensorShape("data");
  EXPECT_EQ(context_input_shape.d[0], static_cast<int>(kMaxBatch));

  // After setting shape, context shape resolves but engine shape stays -1
  EXPECT_TRUE(model->SetBindingShape("data", nvinfer1::Dims4{2, 3, 224, 224}));
  auto resolved_shape = model->GetTensorShape("data");
  EXPECT_EQ(resolved_shape.d[0], 2);

  auto still_dynamic_engine_shape = model->GetEngineTensorShape("data");
  EXPECT_EQ(still_dynamic_engine_shape.d[0], -1);

  // Name queries
  auto input_names = model->GetInputNames();
  auto output_names = model->GetOutputNames();
  EXPECT_EQ(input_names.size(), 1);
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(input_names[0], "data");
  EXPECT_EQ(output_names[0], "resnetv17_dense0_fwd");

  EXPECT_TRUE(model->HasInputBinding("data"));
  EXPECT_TRUE(model->HasOutputBinding("resnetv17_dense0_fwd"));
  EXPECT_FALSE(model->HasInputBinding("resnetv17_dense0_fwd"));
}

TEST(ModelInferenceStandaloneTest, NewApiNotLoadedTest) {
  TensorRTEngine engine("nonexistent.engine");
  EXPECT_FALSE(engine.IsLoaded());

  // All query APIs should return safe defaults on uninitialized engine
  EXPECT_TRUE(engine.GetInputNames().empty());
  EXPECT_TRUE(engine.GetOutputNames().empty());
  EXPECT_FALSE(engine.HasInputBinding("anything"));
  EXPECT_FALSE(engine.HasOutputBinding("anything"));
  EXPECT_EQ(engine.GetCudaStream(), nullptr);
  EXPECT_TRUE(engine.Synchronize());  // no-op on null stream

  auto shape = engine.GetEngineTensorShape("anything");
  EXPECT_EQ(shape.nbDims, -1);
}

TEST_F(ModelInferenceTest, ConstructorOutputAllocationTest) {
  const std::string kInputName = "data";
  const std::string kOutputName = "resnetv17_dense0_fwd";
  constexpr uint32_t kOutputLen = 1000;
  constexpr uint32_t kMaxBatch = 4;

  auto model = LoadEngine();
  if (!model || !model->IsLoaded()) {
    GTEST_SKIP() << "Engine not available, skip ConstructorOutputAllocationTest";
  }

  auto output_tensor = model->GetBindingByName(kOutputName);
  ASSERT_NE(output_tensor, nullptr);

  EXPECT_EQ(output_tensor->byte_size(), kMaxBatch * kOutputLen * sizeof(float));
  EXPECT_EQ(output_tensor->num_elements(), kMaxBatch * kOutputLen);
  EXPECT_GE(output_tensor->capacity_bytes(), kMaxBatch * kOutputLen * sizeof(float));

  auto out_shape = model->GetTensorShape(kOutputName);
  EXPECT_EQ(out_shape.nbDims, 2);
  EXPECT_EQ(out_shape.d[0], static_cast<int>(kMaxBatch));
  EXPECT_EQ(out_shape.d[1], static_cast<int>(kOutputLen));
}

TEST_F(ModelInferenceTest, DynamicOutputCapacityTest) {
  const std::string kInputName = "data";
  const std::string kOutputName = "resnetv17_dense0_fwd";
  constexpr uint32_t kInputH = 224;
  constexpr uint32_t kInputW = 224;
  constexpr uint32_t kOutputLen = 1000;
  constexpr uint32_t kMaxBatch = 4;

  auto model = LoadEngine();
  if (!model || !model->IsLoaded()) {
    GTEST_SKIP() << "Engine not available, skip DynamicOutputCapacityTest";
  }

  const size_t kMaxOutputBytes = kMaxBatch * kOutputLen * sizeof(float);
  auto input_tensor = model->GetBindingByName(kInputName);
  auto output_tensor = model->GetBindingByName(kOutputName);
  ASSERT_NE(input_tensor, nullptr);
  ASSERT_NE(output_tensor, nullptr);

  EXPECT_GE(output_tensor->capacity_bytes(), kMaxOutputBytes);

  // batch = 1
  EXPECT_TRUE(model->SetBindingShape(kInputName, nvinfer1::Dims4{1, 3, kInputH, kInputW}));
  EXPECT_TRUE(input_tensor->Reshape(nvinfer1::Dims4{1, 3, kInputH, kInputW}));
  std::vector<float> input_data(1 * 3 * kInputH * kInputW, 1.0f);
  EXPECT_TRUE(input_tensor->CopyFromHostAsync(input_data.data(), input_tensor->stream()));
  EXPECT_TRUE(model->Infer());

  std::vector<float> output_data(1 * kOutputLen);
  EXPECT_TRUE(model->GetOutputData(kOutputName, output_data.data(), output_data.size() * sizeof(float), false));
  EXPECT_EQ(output_tensor->byte_size(), 1 * kOutputLen * sizeof(float));
  EXPECT_GE(output_tensor->capacity_bytes(), kMaxOutputBytes);

  // batch = max
  EXPECT_TRUE(model->SetBindingShape(kInputName,
                                     nvinfer1::Dims4{static_cast<int64_t>(kMaxBatch), 3, kInputH, kInputW}));
  EXPECT_TRUE(input_tensor->Reshape(
      nvinfer1::Dims4{static_cast<int64_t>(kMaxBatch), 3, kInputH, kInputW}));
  std::vector<float> input_data_max(kMaxBatch * 3 * kInputH * kInputW, 1.0f);
  EXPECT_TRUE(input_tensor->CopyFromHostAsync(input_data_max.data(), input_tensor->stream()));
  EXPECT_TRUE(model->Infer());

  std::vector<float> output_data_max(kMaxBatch * kOutputLen);
  EXPECT_TRUE(model->GetOutputData(kOutputName, output_data_max.data(), output_data_max.size() * sizeof(float), false));
  EXPECT_EQ(output_tensor->byte_size(), kMaxOutputBytes);
}

}  // namespace tensorrt_engine
