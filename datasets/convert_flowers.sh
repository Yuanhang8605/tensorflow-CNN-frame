#!/bin/bash

python convert_dataset_to_tfrecord.py     \
        --dataset_dir=./flowers/flower_photos   \
        --output_directory=./flowers   \
        --train_shards=4   \
        --validation_shards=4   \
        --num_threads=4   \
        --image_format=jpg   \
        --labels_file=./flowers/flower_photos/labels.txt