# Coyright 2018 The TensorFlow Author YuanLi. All Rights Reserved. 

# ==============================================================================
"""Contains utilities for encode images to tfrecord file. 

suitable for tensorflow 1.5 or higher edition
for low edition, see the tfmodelzoo/tutorials/image/cifar10-estimator
  example to modify the tf.data API to tf.contrib.data

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# sys.path.append('.')
# from preprocess_utils import apply_with_random_selector
# from preprocess_utils import decode_and_crop
# from preprocess_utils import distort_color
# from preprocess_utils import mean_image_subtraction

from .preprocess_utils import apply_with_random_selector
from .preprocess_utils import decode_and_crop
from .preprocess_utils import distort_color
from .preprocess_utils import mean_image_subtraction


class Image_PipeLine_Builder(object):
  """ class for build batch image pipeline for CNN"""
  def __init__(self, 
                data_dir, 
                output_image_shape,
                fast_mode = True,
                seed = 2222,
                add_image_summary = True
                ):
    """ctor for Image_PipeLine_Builder class
    
    Args:
      data_dir: a string,  the directory of the dataset
      output_image_shape: a list [height, weight], represent
                          the pipeline output image shape
      fast_mode: a bool, whether use fast mode
      seed: a integer, seed for random shuffle op
      add_image_summary: a bool, whether add image summary
    """
    if not os.path.isdir(data_dir):
      raise ValueError('%s does not exist!' % data_dir)
    if data_dir.endswith('/'):
      data_dir = data_dir[:-1]
    self._data_dir = data_dir
    self._output_image_shape = output_image_shape
    self._fast_mode = fast_mode
    self._seed = seed
    self._add_image_summary = add_image_summary


  @property
  def image_shape(self):
    return self._output_image_shape
  

  def set_image_shape(self, value):
    if not isinstance(value, (list, tuple)):
      raise ValueError('image_shape should be a list or tuple [height, width]')
    if len(value) != 2:
      raise ValueError('image shape must be [height, width] style.')
    self._output_image_shape = value


  @property
  def fast_mode(self):
    return self._fast_mode


  def set_fast_mode(self, value):
    if not isinstance(value, bool):
      raise ValueError('fast_mode must be bool. ')
    self._fast_mode = value


  def _example_parser(self, example_proto):
    """helpler func for parse a tf example"""
    features = {
      'image/encoded': tf.FixedLenFeature(
        (), tf.string, default_value=""),
      'image/class/label': tf.FixedLenFeature(
        (), tf.int64, default_value=tf.zeros([], dtype=tf.int64))}

    parsed_features = tf.parse_single_example(example_proto, features)
    return (parsed_features["image/encoded"], 
            parsed_features["image/class/label"])


  def _tf_parser_for_train(self, example_proto):
    """tensorflow example proto parser func for train
  
    Args:
      example_proto: A Tensor of type 'string' contain many serilized example proto object.
    
    return:
      A tuple Tensor of the image data. 
    """
    with tf.name_scope('train_images'):
      # parser the example
      image, label = self._example_parser(example_proto)
      # preprocess the image
      # decode and crop the image
      dis_image = decode_and_crop(image)
      
      if dis_image.dtype != tf.float32:
        # dis_image = tf.image.convert_image_dtype(dis_image, tf.float32)
        dis_image = tf.to_float(dis_image)

      # resize image
      num_resize_cases = 1 if self._fast_mode else 4
      dis_image = apply_with_random_selector(
          dis_image,
          lambda x, method: tf.image.resize_images(x, self._output_image_shape, method),
          num_cases=num_resize_cases)

      # Randomly flip the image horizontally
      dis_image = tf.image.random_flip_left_right(dis_image)

      # Randomly distort the colors. There are 4 ways to do it.
      dis_image = apply_with_random_selector(
          dis_image,
          lambda x, ordering: distort_color(x, ordering, self._fast_mode),
          num_cases=4)

      if self._add_image_summary:
        tf.summary.image('train_processed_image', 
                          tf.expand_dims(dis_image, 0))
      means = [123.68, 116.78, 103.94]
      dis_image = mean_image_subtraction(dis_image, means)
      # dis_image = tf.subtract(dis_image, 0.5)
      # dis_image = tf.multiply(dis_image, 2.0)
      label = tf.cast(label, tf.int32)
      return dis_image, label 


  def _tf_parser_for_validation(self, example_proto):
    """tensorflow example proto parser func for validation
    
    Args:
      example_proto: A Tensor of type 'string' contain many serilized example proto object.
      use_distortion: bool, for train mode is True, else False 
    
    return:
      A tuple Tensor of the image data. 
    """
    with tf.name_scope('validation_images'):
      # parser the example
      image, label = self._example_parser(example_proto)
      image = tf.image.decode_jpeg(image, channels=3)
      # image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      image = tf.to_float(image)
      if self._use_central_fraction:
        image = tf.image.central_crop(image, central_fraction=self._central_fraction_rate)
      # Resize the image to the specified height and width
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, self._output_image_shape,
                                        align_corners=False)
      image = tf.squeeze(image, [0])
      means = [123.68, 116.78, 103.94]
      image = mean_image_subtraction(image, means)
      # image = tf.subtract(image, 0.5)
      # image = tf.multiply(image, 2.0)
      label = tf.cast(label, tf.int32)
      return image, label


  def build_images_and_labels(self, mode, batch_size, 
                              prefetch_buffer_size,
                              num_parallel_calls= 4,
                              use_central_fraction= True,
                              central_fraction_rate = 0.875):
    """ build the input sample for CNN

    Args:
      mode: 'train' or 'validation'
      batch_size: batch_size for SGD
      prefetch_buffer_size: buffers for feed images to CNN, commonly 2 * batch_size
      num_parallel_calls: parallel threads to preprocess the images. 
                          usually the numbers of cpu core.
    
    return:
      A tuple Tensor (images, labels)
    """
    with tf.device('/cpu:0'):
      if mode == 'train':
        pattern = '*train*.tfrecord'
      else:
        pattern = '*validation*.tfrecord'
        self._use_central_fraction = use_central_fraction
        self._central_fraction_rate = central_fraction_rate

      # Glob to get the tfreord files. 
      tfr_files = tf.gfile.Glob('%s/%s' % (self._data_dir, pattern))
      dataset = tf.data.TFRecordDataset(tfr_files)

      # dataset = dataset.apply(tf.contrib.data.map_and_batch(
      #          map_func= _tf_parser_func, batch_size= FLAGS.batch_size))
      if mode == 'train':
        dataset = dataset.map(self._tf_parser_for_train, 
                        num_parallel_calls= num_parallel_calls)
      else:
        dataset = dataset.map(self._tf_parser_for_validation, 
                        num_parallel_calls= num_parallel_calls)
      dataset.shuffle(batch_size * 3, seed=self._seed)
      dataset.repeat()
      dataset = dataset.batch(batch_size)
      dataset = dataset.prefetch(buffer_size= prefetch_buffer_size)
      iterator = dataset.make_one_shot_iterator()
      image_batch, label_batch = iterator.get_next()    
      return image_batch, label_batch


def main(_):
  """ Test the input image pipeline"""
  data_dir = './flowers'

  pl_builder = Image_PipeLine_Builder(data_dir, [224, 224])

  mode = 'train'
  batch_size = 32
  prefetch_buffer_size = 2 * batch_size

  images, labels = pl_builder.build_images_and_labels(
                                     mode, batch_size,
                                     prefetch_buffer_size,
                                     num_parallel_calls=4)

  with tf.Session() as sess:
    run_op = [images, labels]
    for i in range(2):
      np_images, np_labels = sess.run(run_op)

  #==================== 
  # Plot images
  #==================== 
  plt.figure()
  plt.hold(True)
  for i in range(8):
    plt.subplot(4,2,i+1)
    plt.imshow(np_images[i,:,:,:]/255.0) 
    plt.title("label num is %d" % 
                        np_labels[i])

  plt.show()  


if __name__ == '__main__':
  tf.app.run()
