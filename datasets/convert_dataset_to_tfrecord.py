from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import math
import tensorflow as tf
import numpy as np

import threading

import random

# sys.path.append('.')
from image_to_tfrecord_utils import convert_to_tfrecord

_RANDOM_SEED = 222
_NUM_VALIDATION = 350


tf.app.flags.DEFINE_string('dataset_dir', '/tmp/',
                           'dataset directory')
tf.app.flags.DEFINE_string('output_directory', '/tmp/',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 4,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 4,
                            'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 4,
                            'Number of threads to preprocess the images.')

tf.app.flags.DEFINE_string('image_format', 'jpg',
                           'Image format, jpg or png or bmp')                        

# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   dog
#   cat
#   flower
# where each line corresponds to a label. We map each label contained in
# the file to an integer corresponding to the line number starting from 0.

#   Alternatively, You can use the shell command to write a labels.txt file yourself:
#   in the dataset directory, execute the command:
#     find -type d | egrep "([a-zA-Z1-9])+" | sed 's/\.\///g' | sort > labels.txt
tf.app.flags.DEFINE_string('labels_file', '', 'Labels file')

FLAGS = tf.app.flags.FLAGS

def _get_image_and_labels(dataset_dir, labels_file, fmt='jpg'):
  """Get all imagefiles and corresponding labels
  
  Args:
    dataset_dir: str, dataset directory
    labels_file: str, a .txt file path record all the class names
    fmt: str, image format

  return:
    filenames:  all images absolutely path
    label_and_names: all label id and names
  """
  with tf.gfile.FastGFile(labels_file, 'r') as f:
    uniq_labels = [l.strip() for l in f.readlines()]
  
  # leave 0 to background label
  filenames = []
  label_and_names = []
  label_id = 1
  for name in uniq_labels:
    pattern = '%s/%s/*.%s' % (dataset_dir, name, fmt)
    files = tf.gfile.Glob(pattern)
    filenames.extend(files)
    labels = [label_id] * len(files)
    names = [name] * len(files)
    label_and_names.extend(zip(labels, names))
    label_id += 1
  
  # random shuffle the data
  shuffled_index = list(range(len(filenames)))
  random.seed(_RANDOM_SEED)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  label_and_names = [label_and_names[i] for i in shuffled_index]

  return filenames, label_and_names


def _process_images_batch(split, thread_id, filenames, label_and_names):
  """threading function to process the images"""
  if split == 'train':
    num_shards = FLAGS.train_shards
  else:
    num_shards = FLAGS.validation_shards
  
  shards_per_thread = int(num_shards / FLAGS.num_threads)
  nums_per_shards = int(math.ceil(len(filenames) / num_shards))
  with tf.Graph().as_default():
    with tf.Session() as sess:
      for i in range(shards_per_thread):
        shard_id = thread_id * shards_per_thread + i
        start = nums_per_shards * shard_id
        end = min(start + nums_per_shards, len(filenames))
        convert_to_tfrecord(sess, split, shard_id, num_shards, 
                filenames[start:end], label_and_names[start:end],
                FLAGS.output_directory, FLAGS.image_format)


def _process_images(split, filenames, label_and_names):
  """Multi-Thread frame to process images"""
  assert split in ['train', 'validation']
  coord = tf.train.Coordinator()
  threads = []
  for thread_id in range(FLAGS.num_threads):
    t = threading.Thread(target=_process_images_batch, args=(split, thread_id, filenames, label_and_names))
    t.start()
    threads.append(t)
  
  coord.join(threads)


def main(_):
  assert not FLAGS.train_shards % FLAGS.num_threads
  assert not FLAGS.validation_shards % FLAGS.num_threads

  filenames, label_and_names = _get_image_and_labels(FLAGS.dataset_dir, FLAGS.labels_file, FLAGS.image_format)
  sys.stdout.write('>>> Converting validation images dataset to tfrecord. \n')
  _process_images('validation', filenames[:_NUM_VALIDATION], label_and_names[:_NUM_VALIDATION])
  sys.stdout.write('>>> Converting train images dataset to tfrecord. \n')
  _process_images('train', filenames[_NUM_VALIDATION:], label_and_names[_NUM_VALIDATION:])


if __name__ == '__main__':
  tf.app.run()