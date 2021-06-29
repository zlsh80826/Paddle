// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

class SwishPlugin : public PluginTensorRT {
 private:
  float beta_;

 public:
  size_t getSerializationSize() const override {
    return getBaseSerializationSize() + SerializedSize(beta_);
  }

  void serialize(void* buffer) const override {
    serializeBase(buffer);
    SerializeValue(&buffer, beta_);
  }

  explicit SwishPlugin(const float beta, const bool with_fp16) : beta_(beta) {
    with_fp16_ = with_fp16;
  }

  // It was used for tensorrt deserialization.
  // It should not be called by users.
  SwishPlugin(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    DeserializeValue(&serialData, &serialLength, &beta_);
  }

  ~SwishPlugin() {}

  int initialize() override;

  SwishPlugin* clone() const override {
    return new SwishPlugin(beta_, with_fp16_);
  }

  const char* getPluginType() const override { return "swish_plugin"; }
  int getNbOutputs() const override { return 1; }
  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                     int nbInputDims) override;
#if IS_TRT_VERSION_LT(8000)
  int enqueue(int batchSize, const void* const* inputs, void** outputs,
#else
  int enqueue(int batchSize, const void* const* inputs, void* const* outputs,
#endif
              void* workspace, cudaStream_t stream) override;
};

class SwishPluginCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const override { return "swish_plugin"; }

  const char* getPluginVersion() const override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length) override {
    return new SwishPlugin(serial_data, serial_length);
  }
};
REGISTER_TRT_PLUGIN_V2(SwishPluginCreator);

#if IS_TRT_VERSION_GE(6000)
class SwishPluginDynamic : public DynamicPluginTensorRT {
 public:
  explicit SwishPluginDynamic(const float beta, const bool with_fp16)
      : beta_(beta) {
    with_fp16_ = with_fp16;
  }
  SwishPluginDynamic(void const* serialData, size_t serialLength) {
    DeserializeValue(&serialData, &serialLength, &beta_);
    DeserializeValue(&serialData, &serialLength, &with_fp16_);
  }
  nvinfer1::IPluginV2DynamicExt* clone() const override {
    return new SwishPluginDynamic(beta_, with_fp16_);
  }

  const char* getPluginType() const override { return "swish_plugin_dynamic"; }
  int getNbOutputs() const override { return 1; }
  int initialize() override;

  size_t getSerializationSize() const override;
  void serialize(void* buffer) const override;

  nvinfer1::DimsExprs getOutputDimensions(
      int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
      nvinfer1::IExprBuilder& expr_builder) override;

  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc* inOut,
                                 int nbInputs, int nbOutputs) override;

  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out,
                       int nbOutputs) override {}

  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc* outputs,
                          int nbOutputs) const override {
    return 0;
  }

  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
              const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) override;
  nvinfer1::DataType getOutputDataType(int index,
                                       const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const override;

  void destroy() override { delete this; }

 private:
  float beta_;
};

class SwishPluginDynamicCreator : public TensorRTPluginCreator {
 public:
  const char* getPluginName() const override { return "swish_plugin_dynamic"; }

  const char* getPluginVersion() const override { return "1"; }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serial_data,
                                         size_t serial_length) override {
    return new SwishPluginDynamic(serial_data, serial_length);
  }
};
REGISTER_TRT_PLUGIN_V2(SwishPluginDynamicCreator);
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
