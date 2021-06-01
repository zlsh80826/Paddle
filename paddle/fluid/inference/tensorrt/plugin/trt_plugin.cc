// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

void PluginTensorRT::serializeBase(void*& buffer) const {
  SerializeValue(&buffer, input_dims_);
  SerializeValue(&buffer, max_batch_size_);
  SerializeValue(&buffer, data_type_);
  SerializeValue(&buffer, data_format_);
  SerializeValue(&buffer, with_fp16_);
}

void PluginTensorRT::deserializeBase(void const*& serial_data,
                                     size_t& serial_length) {
  DeserializeValue(&serial_data, &serial_length, &input_dims_);
  DeserializeValue(&serial_data, &serial_length, &max_batch_size_);
  DeserializeValue(&serial_data, &serial_length, &data_type_);
  DeserializeValue(&serial_data, &serial_length, &data_format_);
  DeserializeValue(&serial_data, &serial_length, &with_fp16_);
}

size_t PluginTensorRT::getBaseSerializationSize() const {
  return (SerializedSize(input_dims_) + SerializedSize(max_batch_size_) +
          SerializedSize(data_type_) + SerializedSize(data_format_) +
          SerializedSize(with_fp16_));
}

bool PluginTensorRT::supportsFormat(nvinfer1::DataType type,
                                    nvinfer1::PluginFormat format) const TRT_NOEXCEPT {
  return ((type == nvinfer1::DataType::kFLOAT) &&
          (format == nvinfer1::PluginFormat::kLINEAR));
}

void PluginTensorRT::configureWithFormat(
    const nvinfer1::Dims* input_dims, int num_inputs,
    const nvinfer1::Dims* output_dims, int num_outputs, nvinfer1::DataType type,
    nvinfer1::PluginFormat format, int max_batch_size) TRT_NOEXCEPT {
  data_type_ = type;
  data_format_ = format;
  input_dims_.assign(input_dims, input_dims + num_inputs);
  max_batch_size_ = max_batch_size;
}

void PluginTensorRT::setPluginNamespace(nvinfer1::AsciiChar const *plugin_namespace) TRT_NOEXCEPT {
  namespace_ = plugin_namespace;
}

nvinfer1::AsciiChar const* PluginTensorRT::getPluginNamespace() const TRT_NOEXCEPT {
  return namespace_;
}

void PluginTensorRT::destroy() TRT_NOEXCEPT {
  delete this;
}

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
