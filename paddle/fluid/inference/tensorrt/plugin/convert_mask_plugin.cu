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

#include <cassert>
#include <cstring>
#include <vector>
#include "paddle/fluid/inference/tensorrt/plugin/convert_mask_plugin.h"
#include "paddle/fluid/inference/tensorrt/plugin/trt_plugin_factory.h"
#include "paddle/fluid/operators/math/math_cuda_utils.h"

namespace paddle {
namespace inference {
namespace tensorrt {
namespace plugin {

// Dynamic Plugin below.
#if IS_TRT_VERSION_GE(6000)

/* This plugin currently converts the matmul output [B, S, S]
to the mask with the bertQKV fused_multihead_attention format */

constexpr size_t threadsPerCta128 = 2 * 2 * 32;
constexpr size_t threadsPerCta384 = 1 * 8 * 32;

constexpr size_t xmmasM128 = 4;
constexpr size_t xmmasM384 = 24;

constexpr size_t packedMaskSize64 = xmmasM128 * threadsPerCta128;
constexpr size_t packedMaskSize96 = xmmasM128 * threadsPerCta128;
constexpr size_t packedMaskSize128 = xmmasM128 * threadsPerCta128;
constexpr size_t packedMaskSize384 = xmmasM384 * threadsPerCta384;

nvinfer1::DimsExprs ConvertMaskPluginDynamic::getOutputDimensions(
    int output_index, const nvinfer1::DimsExprs* inputs, int nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) {
  assert(output_index == 0);
  constexpr int BDIM = 0;
  constexpr int SDIM = 1;

  if (type_ == nvinfer1::DataType::kHALF) {
    auto cms64 = expr_builder.constant(packedMaskSize64);
    auto cms96 = expr_builder.constant(packedMaskSize96);
    auto cms128 = expr_builder.constant(packedMaskSize128);
    auto cms384 = expr_builder.constant(packedMaskSize384);
    auto c64 = expr_builder.constant(64);
    auto c96 = expr_builder.constant(96);
    auto c128 = expr_builder.constant(128);
    auto c384 = expr_builder.constant(384);

    auto is64 = expr_builder.operation(nvinfer1::DimensionOperation::kEQUAL,
                                       *inputs[0].d[SDIM], *c64);
    auto is96 = expr_builder.operation(nvinfer1::DimensionOperation::kEQUAL,
                                       *inputs[0].d[SDIM], *c96);
    auto is128 = expr_builder.operation(nvinfer1::DimensionOperation::kEQUAL,
                                        *inputs[0].d[SDIM], *c128);
    auto is384 = expr_builder.operation(nvinfer1::DimensionOperation::kEQUAL,
                                        *inputs[0].d[SDIM], *c384);
    auto sel64 = expr_builder.operation(nvinfer1::DimensionOperation::kPROD,
                                        *is64, *cms64);
    auto sel96 = expr_builder.operation(nvinfer1::DimensionOperation::kPROD,
                                        *is96, *cms96);
    auto sel128 = expr_builder.operation(nvinfer1::DimensionOperation::kPROD,
                                         *is128, *cms128);
    auto sel384 = expr_builder.operation(nvinfer1::DimensionOperation::kPROD,
                                         *is384, *cms384);
    auto maskSize1 = expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                            *sel64, *sel96);
    auto maskSize2 = expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                            *sel384, *sel128);
    auto maskSize = expr_builder.operation(nvinfer1::DimensionOperation::kSUM,
                                           *maskSize1, *maskSize2);
    auto fp16maskSize =
        expr_builder.operation(nvinfer1::DimensionOperation::kPROD, *maskSize,
                               *expr_builder.constant(2));

    nvinfer1::DimsExprs ret;
    ret.nbDims = 2;
    ret.d[0] = inputs[0].d[BDIM];
    ret.d[1] = fp16maskSize;
    return ret;
  }
  nvinfer1::DimsExprs ret;
  ret.nbDims = 1;
  ret.d[0] = inputs[0].d[0];
  return ret;
}

size_t ConvertMaskPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs, int nb_inputs,
    const nvinfer1::PluginTensorDesc* outputs, int nb_outputs) const {
  return inputs[0].dims.d[0] * inputs[0].dims.d[1];
}

bool ConvertMaskPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* in_out, int nb_inputs,
    int nb_outputs) {
  const nvinfer1::PluginTensorDesc& desc = in_out[pos];
  /*  input: [B, S, 1] */
  /* output: [B, 2*maskSize] */
  assert(nb_inputs == 1);
  assert(nb_outputs == 1);

  if (pos == 0) {
    if (type_ == nvinfer1::DataType::kHALF)
      return desc.type == nvinfer1::DataType::kHALF && desc.dims.nbDims == 3;
    return desc.type == nvinfer1::DataType::kFLOAT && desc.dims.nbDims == 3;
  }
  // return true;
  /* fp16 -> fp16, fp32 -> int32 */
  if (type_ == nvinfer1::DataType::kHALF)
    return desc.type == nvinfer1::DataType::kHALF;
  return desc.type == nvinfer1::DataType::kINT32;
}

nvinfer1::DataType ConvertMaskPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType* input_types, int nb_inputs) const {
  PADDLE_ENFORCE_EQ(index, 0,
                    platform::errors::InvalidArgument(
                        "The convert mask plugin only has one input, so the "
                        "index value should be 0, but get %d.",
                        index));
  if (type_ == nvinfer1::DataType::kHALF) {
    return nvinfer1::DataType::kHALF;
  }
  return nvinfer1::DataType::kINT32;
}

/* half [B, S, 1] -> int [S, B, 1] */
template <typename T>
__global__ void FullMaskPreprocess(const T* input, int* output, int seq_len,
                                   int batch) {
  int bid = blockIdx.x;
  int sid = threadIdx.x;
  output[sid * batch + bid] = static_cast<int>(input[bid * seq_len + sid]);
}

/* float [B, S, 1] -> int [B] */
/* [[1. 1. 1. 0. 0.], -> [3, 4]
    [1. 1. 1. 1. 0.]]           */
__global__ void IMaskPreprocess(const float* input, int* output, int seq_len,
                                int batch) {
  float sum = 0.f;
  int bid = blockIdx.x;
  int sid = threadIdx.x;
  float thread_data = input[bid * seq_len + sid];

  sum = paddle::operators::math::blockReduceSum<float>(thread_data, 0xffffffff);

  if (sid == 0) {
    output[bid] = static_cast<int>(sum);
  }
}

__global__ void fillSBSMaskKernel(const uint32_t warps_m,
                                  const uint32_t warps_n, const uint32_t S,
                                  const int* inputMaskSB,
                                  uint32_t* inputMaskX) {
  extern __shared__ int shm_mask[];  // S mask elements of this batch

  const size_t xmmas_n = (S + 16 * warps_n - 1) / (16 * warps_n);
  const uint32_t threads_per_cta = blockDim.x;
  const uint32_t xmmas_m = gridDim.x;
  const uint32_t B = gridDim.y;

  const uint32_t mi = blockIdx.x;
  const uint32_t bi = blockIdx.y;
  const uint32_t tidx = threadIdx.x;

  const size_t warp = tidx / 32;
  const size_t warp_m = warp % warps_m;
  const size_t warp_n = warp / warps_m;
  const size_t lane = tidx % 32;
  const size_t col = warp_n * 16 + lane % 4 * 2;

  // load the mask corresponding to one batch
  for (uint32_t si = tidx; si < S; si += threads_per_cta) {
    // not coalesced to conform to current input format: SxB
    shm_mask[si] = inputMaskSB[si * B + bi];
  }
  __syncthreads();

  uint32_t mask = 0u;

  for (size_t ni = 0; ni < xmmas_n; ++ni) {
    const int offset = ni * 16 * warps_n + col;
    mask |= (shm_mask[offset + 0] == 1.f ? 1u : 0u) << (8 * ni + 0);
    mask |= (shm_mask[offset + 1] == 1.f ? 1u : 0u) << (8 * ni + 1);
    mask |= (shm_mask[offset + 0] == 1.f ? 1u : 0u) << (8 * ni + 2);
    mask |= (shm_mask[offset + 1] == 1.f ? 1u : 0u) << (8 * ni + 3);
    mask |= (shm_mask[offset + 8] == 1.f ? 1u : 0u) << (8 * ni + 4);
    mask |= (shm_mask[offset + 9] == 1.f ? 1u : 0u) << (8 * ni + 5);
    mask |= (shm_mask[offset + 8] == 1.f ? 1u : 0u) << (8 * ni + 6);
    mask |= (shm_mask[offset + 9] == 1.f ? 1u : 0u) << (8 * ni + 7);
  }

  inputMaskX[(bi * xmmas_m + mi) * threads_per_cta + tidx] = mask;
}

void convertMask(const uint32_t S, const uint32_t B, const uint32_t warps_m,
                 const uint32_t warps_n, const uint32_t warps_k,
                 const int* inputMaskSB, uint32_t* inputMaskX,
                 cudaStream_t stream) {
  const size_t xmmas_m = (S + 16 * warps_m - 1) / (16 * warps_m);

  const size_t threads_per_cta = warps_m * warps_n * warps_k * 32;
  dim3 grid(xmmas_m, B);
  fillSBSMaskKernel<<<grid, threads_per_cta, S * sizeof(int), stream>>>(
      warps_m, warps_n, S, inputMaskSB, inputMaskX);
}

int ConvertMaskPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) {
  auto input_dims = input_desc[0].dims;
  auto output_dims = output_desc[0].dims;
  size_t num_elements = ProductDim(input_dims);
  size_t out_num_elements = ProductDim(output_dims);
  int batch = input_dims.d[0];
  int seq_len = input_dims.d[1];

  if (type_ == nvinfer1::DataType::kFLOAT) {
    IMaskPreprocess<<<batch, seq_len, 0, stream>>>(
        static_cast<const float*>(inputs[0]), static_cast<int*>(outputs[0]),
        seq_len, batch);
  } else {
    int* inputMaskSB = reinterpret_cast<int*>(workspace);
    FullMaskPreprocess<half><<<batch, seq_len, 0, stream>>>(
        static_cast<const half*>(inputs[0]), inputMaskSB, seq_len, batch);
    size_t warps_m = 0, warps_n = 0, warps_k = 1;
    if (seq_len == 64 || seq_len == 96 || seq_len == 128) {
      warps_m = 2;
      warps_n = 2;
    } else if (seq_len == 384) {
      warps_m = 1;
      warps_n = 8;
    } else {
      assert(false);
    }
    convertMask(seq_len, batch, warps_m, warps_n, warps_k, inputMaskSB,
                static_cast<uint32_t*>(outputs[0]), stream);
  }

  return cudaGetLastError() != cudaSuccess;
}
#endif

}  // namespace plugin
}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
