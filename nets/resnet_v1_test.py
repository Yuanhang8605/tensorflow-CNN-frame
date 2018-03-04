# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for slim.nets.resnet_v1."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import sys
sys.path.append('..')

from nets import resnet_utils
from nets import resnet_v1
from datasets.build_image_pipeline import Image_PipeLine_Builder

slim = tf.contrib.slim


def _create_test_input(data_dir, output_image_shape):
  """Create test input tensor.

  Args:
    data_dir: A string represent the data directory
    output_image_shape: A [height, width] list represent image shape

  Returns:
    images, labels: Tensor pipeline input to the CNN
  """
  pl_builder = Image_PipeLine_Builder(data_dir, output_image_shape)

  mode = 'train'
  batch_size = 32
  prefetch_buffer_size = 2 * batch_size

  images, labels = pl_builder.build_images_and_labels(
                                     mode, batch_size,
                                     prefetch_buffer_size,
                                     num_parallel_calls=4)
  return images, labels                                    


class ResnetTest(tf.test.TestCase):
  """ Class used to test the resnet"""
  def test_resnet(self):
    with self.test_session() as sess:
      data_dir = '../datasets/flowers'  
      image_shape = [224, 224]
      images, labels = _create_test_input(data_dir, image_shape)
      with slim.arg_scope(resnet_utils.resnet_arg_scope()):
        net, endo_points = resnet_v1.resnet_v1_50(images, num_classes=5)
      tf.global_variables_initializer().run()
      self.assertTrue(net.op.name.startswith('resnet_v1_50/SpatialSqueeze'))
      prediction = net.eval()
      # self.assertListEqual(prediction.shape, (32, 5))
      self.assertEqual(prediction.shape, (32,5))

if __name__ == '__main__':
  tf.test.main()
