# Coyright 2018 The TensorFlow Author YuanLi. All Rights Reserved. 

# ==============================================================================
"""Contains utilities for encode images to tfrecord file. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf
import numpy as np

#======================================
#  image format transfer: from png to jpg
#======================================
def png_to_jpg(filename):
  if '.png' in filename:
    return
  with tf.gfile.FastGFile(filename, 'rb') as f:
    buffer = f.read()
  image = tf.image.decode_png(buffer, channels=3)
  return tf.image.encode_jpeg(image, format='rgb', quality=100)       #output a string buffer


#======================================
#  image Reader and process
#======================================
class ImageDecoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._buffer = tf.placeholder(dtype=tf.string)
    self._fmt = 'jpg'
  
  def set_image_format(self, fmt):
    """ set image format, defautly is jpg"""
    assert fmt in ['jpg', 'png', 'bmp']
    self._fmt = fmt

  def get_image_shape(self, sess, image_buffer):
    """ get image shape."""
    if self._fmt == 'jpg':
      # because only parse the header of imagebuffer, it's much faster than the decode method
      image_shape = tf.image.extract_jpeg_shape(self._buffer)          
      image_shape = sess.run(image_shape, feed_dict={self._buffer: image_buffer})
      return image_shape[0], image_shape[1]

    image = self.decode_image(sess, image_buffer)
    return image.shape[0], image.shape[1]
  
  def decode_image(self, sess, image_buffer):
    if self._fmt == 'jpg':
      decode_img = tf.image.decode_jpeg(self._buffer, channels=3)
    elif self._fmt == 'png':
      decode_img = tf.image.decode_png(self._buffer, channels=3)
    elif self._fmt == 'bmp':
      decode_img = tf.image.decode_bmp(self._buffer, channels=3)

    image = sess.run(decode_img, feed_dict={self._buffer: image_buffer})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


#======================================
#  TFRecord features wrapper related 
#======================================
def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
  """Returns a TF-Feature of floats.

  Args:
    values: A scalar of list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


#======================================
#  transfer image to tfexample 
#======================================
def image_to_tfexample(image_data, image_format, height, width, class_label, class_name):
  """ transfer image str buffer and other info to tfexample protol.

  Args:
    image_data: str buffer of image. using tf.gfile.FastGfile(file).read to get
    image_format: a str
    height: an integer, represent image height
    width: an integer, represent image width
    class_label: an integer, represent class label 
    class_name : a str, represent class name
  
  return:
    a tf.example proto object
  """
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_label),
      'image/class/name' : bytes_feature(class_name),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))


#======================================
# Writing tfexample to TFRecord format 
#======================================
def convert_to_tfrecord(sess, split_name, shard_id, num_shards, filenames, 
                  class_label_and_names, output_directory, fmt=b'jpg'):
  """Serialize jpg file to tf example proto file. 

  Args:
    sess: tensorflow Session to run the op
    split_name:  either 'train' or 'validation'
    threadidx:  the idx of the thread
    shard_id:   the idx of the shards
    filenames: A list of abs paths to jpg or png images
    class_label_and_names: A list contain (label, class_name)
    output_directory: where to save the tfrecord file
    fmt: the image format
  """
  
  assert split_name in ['train', 'validation']
  output_filename = '%s/%s_%05d-of-%05d.tfrecord' % (
          output_directory, split_name, shard_id, num_shards) 
  sys.stdout.write('\r>> Coverting images to %s' % output_filename)
  sys.stdout.write('\n')
  # with tf.Graph().as_default():
  image_decoder = ImageDecoder()
  idx = 0
  # with tf.Session() as sess:
  with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
    for fname in filenames:
      try: 
        with tf.gfile.FastGFile(fname, mode='rb') as f:
          buffer = f.read()
        image_decoder.set_image_format(fmt)
        height, width = image_decoder.get_image_shape(sess, buffer)
      except Exception as e:
        print(e)
        print('SKIPPED: Unexpected eror while decoding %s.' % fname)
        continue
        
      class_label, class_name = class_label_and_names[idx]
      tfexample = image_to_tfexample(buffer, fmt, height, width, 
                                      class_label, class_name)
      tfrecord_writer.write(tfexample.SerializeToString())
      idx += 1

  sys.stdout.flush()
# #======================================
# #  dataset glob utilities
# #======================================
# """
# Commonly, a dataset should apply a labels.txt to identify every label class names. 
# Typically dataset structure:
#     dataset_dir/
#       label1/, label2/, ...
#       ...
  
#   Alternatively, You can use the shell command to write a labels.txt file yourself
#     find -type d | egrep "([a-zA-Z1-9])+" | sed 's/\.\///g' > labels.txt

# """

# def read_label_file(dataset_dir, filename=LABELS_FILENAME):
#   """Reads the labels file and returns a mapping from ID to class name.

#   Args:
#     dataset_dir: The directory in which the labels file is found.
#     filename: The filename where the class names are written.

#   Returns:
#     A map from a class name to label (integer).
#   """
#   labels_filename = os.path.join(dataset_dir, filename)
#   print('Obtain labels_to_classnames dict from %s.' % labels_filename)
#   unique_class_names = [l.strip() for l in tf.gfile.FastGFile(
#       labels_filename, 'r').readlines()]

#   classnames_to_labels = {}
#   # Leave label index 0 empty as a background class
#   label_index = 1
#   for name in unique_class_names:
#     classnames_to_labels[name] = label_index
#     label_index += 1
#   return classnames_to_labels


# #======================================
# #  glob file and get the filenames and classes.  
# #======================================
# def get_filenames_and_labels(dataset_dir, classnames_to_labels, fmt='jpg'):
#   """Returns a list of filenames and inferred label_class names."""

#   filenames = []
#   label_class_list = []

#   match_names = tf.gfile.ListDirectory(dataset_dir)
#   for name in match_names:
#     # file match 
#     search_results = [(i in name) for i in classnames_to_labels]
#     if not any(search_results):
#       continue
#     path = os.path.join(dataset_dir, name)
#     if not tf.gfile.IsDirectory(path):
#       continue
#     img_files = tf.gfile.Glob('%s/*.%s' % (path, fmt))
#     filenames.extend(img_files)


# if __name__ == '__main__':
#   #  test the ImageDecoder class 
#   with tf.gfile.FastGFile('test.jpg', 'rb') as f:
#     buffer = f.read()
#   decoder = ImageDecoder()
#   sess = tf.Session()
#   print('image shape is: {!r}'.format(
#                 decoder.get_image_shape(sess, buffer)))
#   image = decoder.decode_image(sess, buffer)
#   print(image)
#   sess.close()










