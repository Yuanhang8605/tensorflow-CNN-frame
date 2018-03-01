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

from preprocess_utils import apply_with_random_selector
from preprocess_utils import decode_and_crop
from preprocess_utils import distort_color


# CNN input image shape
_INPUT_IMG_SHAPE = [299, 299]
# Random seed 
_SEED = 2222
# whether apply central_fraction in validation preprocess
_IS_CENTRAL_FRACTION = True
_CENTRAL_FRACTION = 0.875
# whether add image summary
_ADD_IMAGE_SUMMARY = True
# whether using fastmode
_FAST_MODE = True 


def _example_parser(example_proto):
  """helpler func for parse a tf example"""
  features = {
    'image/encoded': tf.FixedLenFeature(
      (), tf.string, default_value=""),
    'image/class/label': tf.FixedLenFeature(
      (), tf.int64, default_value=tf.zeros([], dtype=tf.int64))}

  parsed_features = tf.parse_single_example(example_proto, features)
  return (parsed_features["image/encoded"], 
          parsed_features["image/class/label"])


def _tf_parser_for_train(example_proto):
  """tensorflow example proto parser func for train
 
  Args:
    example_proto: A Tensor of type 'string' contain many serilized example proto object.
  
  return:
    A tuple Tensor of the image data. 
  """
  with tf.name_scope('train_images'):
    # parser the example
    image, label = _example_parser(example_proto)
    # preprocess the image
    # decode and crop the image
    dis_image, dis_bbox = decode_and_crop(image)
    # see the image_with_box, for debug purpose
    image_with_box = tf.image.decode_jpeg(image, channels=3)
    image_with_box = tf.image.convert_image_dtype(image_with_box,
                                            tf.float32)
    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image_with_box, 0), 
                                                  dis_bbox)
    image_with_box = tf.squeeze(image_with_box, [0])

    # sess = tf.Session()
    # np_image_with_box = sess.run(image_with_box)

    # plt.imshow(np_image_with_box)
    # plt.show()

    if _ADD_IMAGE_SUMMARY:
      tf.summary.image('image_with_box', 
                        tf.expand_dims(image_with_box, 0))
    
    if dis_image.dtype != tf.float32:
      dis_image = tf.image.convert_image_dtype(dis_image, tf.float32)

    # resize image
    num_resize_cases = 1 if _FAST_MODE else 4
    dis_image = apply_with_random_selector(
        dis_image,
        lambda x, method: tf.image.resize_images(x, _INPUT_IMG_SHAPE, method),
        num_cases=num_resize_cases)

    # Randomly flip the image horizontally
    dis_image = tf.image.random_flip_left_right(dis_image)

    # Randomly distort the colors. There are 4 ways to do it.
    dis_image = apply_with_random_selector(
        dis_image,
        lambda x, ordering: distort_color(x, ordering, _FAST_MODE),
        num_cases=4)

    if _ADD_IMAGE_SUMMARY:
      tf.summary.image('train_processed_image', 
                        tf.expand_dims(dis_image, 0))
    # dis_image = tf.subtract(dis_image, 0.5)
    # dis_image = tf.multiply(dis_image, 2.0)
    label = tf.cast(label, tf.int32)
    return dis_image, label 


def _tf_parser_for_validation(example_proto):
  """tensorflow example proto parser func for validation
  
  Args:
    example_proto: A Tensor of type 'string' contain many serilized example proto object.
    use_distortion: bool, for train mode is True, else False 
  
  return:
    A tuple Tensor of the image data. 
  """
  with tf.name_scope('validation_images'):
    # parser the example
    image, label = _example_parser(example_proto)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    if _CENTRAL_FRACTION:
      image = tf.image.central_crop(image, central_fraction=_CENTRAL_FRACTION)
    # Resize the image to the specified height and width
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, _INPUT_IMG_SHAPE,
                                      align_corners=False)
    image = tf.squeeze(image, [0])
    # image = tf.subtract(image, 0.5)
    # image = tf.multiply(image, 2.0)
    label = tf.cast(label, tf.int32)
    return image, label


def build_images_and_labels(mode, data_dir, batch_size,
                          prefetch_buffer_size, num_parallel_calls=4):
  """ build the input sample for CNN

  Args:
    mode: 'train' or 'validation'
    data_dir: abs path of tfrecord files 
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

    # Glob to get the tfreord files. 
    tfr_files = tf.gfile.Glob('%s/%s' % (data_dir, pattern))
    dataset = tf.data.TFRecordDataset(tfr_files)

    # dataset = dataset.apply(tf.contrib.data.map_and_batch(
    #          map_func= _tf_parser_func, batch_size= FLAGS.batch_size))
    if mode == 'train':
      dataset = dataset.map(_tf_parser_for_train, 
                      num_parallel_calls= num_parallel_calls)
    else:
      dataset = dataset.map(_tf_parser_for_validation, 
                      num_parallel_calls= num_parallel_calls)
    dataset.shuffle(batch_size * 3, seed=_SEED)
    dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size= prefetch_buffer_size)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()    
    return image_batch, label_batch


def main(_):
  """ Test the input image pipeline"""
  mode = 'train'
  data_dir = './flowers'
  batch_size = 32
  prefetch_buffer_size = 2 * batch_size

  images, labels = build_images_and_labels(
                                     mode, data_dir, batch_size,
                                     prefetch_buffer_size)

  with tf.Session() as sess:
    run_op = [images, labels]
    np_images, np_labels = sess.run(run_op)

  #==================== 
  # Plot images
  #==================== 
  plt.figure()
  plt.hold(True)
  for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(np_images[i,:,:,:]) 
    plt.title("label num is %d" % 
                        np_labels[i])
  plt.show()  

if __name__ == '__main__':
  tf.app.run()