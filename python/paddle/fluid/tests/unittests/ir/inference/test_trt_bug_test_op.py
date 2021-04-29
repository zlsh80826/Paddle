# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import itertools
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.core import PassVersionChecker
from paddle.fluid.core import AnalysisConfig


class TensorRTBugTest(InferencePassTest):
    def setUp(self):
        with fluid.program_guard(self.main_program, self.startup_program):
            data0 = fluid.data(
                name='data0', shape=[-1, 32, -1, -1], dtype='float32')
            data1 = fluid.data(
                name='data1', shape=[-1, 32, -1, -1], dtype='float32')
            data2 = fluid.data(
                name='data2', shape=[-1, 32, -1, -1], dtype='float32')
            data3 = fluid.data(
                name='data3', shape=[-1, 32, -1, -1], dtype='float32')
            tmp = fluid.layers.elementwise_add(
                x=data0, y=data1, name='elementwise_add_0')
            tmp = fluid.layers.elementwise_add(
                x=tmp, y=data2, name='elementwise_add_1')
            tmp = fluid.layers.elementwise_add(
                x=tmp, y=data3, name='elementwise_add_2')
            out = fluid.layers.batch_norm(tmp, is_test=True)

        self.feeds = self.set_feeds()
        self.enable_trt = True
        self.trt_parameters = TensorRTBugTest.TensorRTParam(
            1 << 30, 1, 1, AnalysisConfig.Precision.Float32, False, False)
        self.dynamic_shape_params = TensorRTBugTest.DynamicShapeParam({
            'data0': [1, 32, 1, 1],
            'data1': [1, 32, 1, 1],
            'data2': [1, 32, 1, 1],
            'data3': [1, 32, 1, 1]
        }, {
            'data0': [1, 32, 128, 128],
            'data1': [1, 32, 64, 64],
            'data2': [1, 32, 64, 64],
            'data3': [1, 32, 64, 64]
        }, {
            'data0': [1, 32, 16, 16],
            'data1': [1, 32, 64, 64],
            'data2': [1, 32, 64, 64],
            'data3': [1, 32, 64, 64]
        }, False)
        self.fetch_list = [out]

    def set_feeds(self):
        return {
            'data0': np.random.random([1, 32, 64, 64]).astype('float32'),
            'data1': np.random.random([1, 32, 64, 64]).astype('float32'),
            'data2': np.random.random([1, 32, 64, 64]).astype('float32'),
            'data3': np.random.random([1, 32, 64, 64]).astype('float32'),
        }

    def check_output(self):
        if core.is_compiled_with_cuda():
            use_gpu = True
            self.check_output_with_option(use_gpu)
            self.assertTrue(
                PassVersionChecker.IsCompatible('tensorrt_subgraph_pass'))

    def test(self):
        self.check_output()


if __name__ == "__main__":
    unittest.main()
