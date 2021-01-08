# Copyright 2016 Google Inc. All Rights Reserved.
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
# ==============================================================================
"""Small library that points to a data set.

Methods of Data class:
  data_files: Returns a python list of all (sharded) data set files.
  num_examples_per_epoch: Returns the number of examples in the data set.
  num_classes: Returns the number of classes in the data set.
  reader: Return a reader for a single entry from the data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
import os

import tensorflow as tf
from tfm_image_processor import CIFAR10Preprocessor


#   This directory should contain two tfrecord files named as train.tfrecords and validation.tfrecords
#   Please follow https://github.com/tensorflow/models/blob/master/research/slim/datasets/download_and_convert_cifar10.py,
#   download the CIFAR-10 dataset, convert it to TFRecord format,
#   and rename the two output files as train.tfrecords and validation.tfrecords.
DATA_PATH = '/home/dingxiaohan/slim_cifar10_tfrecords'


class Dataset(object):
    """A simple class for handling data sets."""
    __metaclass__ = ABCMeta

    def __init__(self, name, subset, data_dir):
        """Initialize dataset using a subset and the path to the data."""
        assert subset in self.available_subsets(), self.available_subsets()
        self.name = name
        self.subset = subset
        self.data_dir = data_dir

    @abstractmethod
    def num_classes(self):
        """Returns the number of classes in the data set."""
        pass
        # return 10

    @abstractmethod
    def num_examples_per_epoch(self):
        """Returns the number of examples in the data subset."""
        pass
        # if self.subset == 'train':
        #   return 10000
        # if self.subset == 'validation':
        #   return 1000

    @abstractmethod
    def download_message(self):
        """Prints a download message for the Dataset."""
        pass

    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'validation']

    @abstractmethod
    def data_files(self):
        """Returns a python list of all (sharded) data subset files.

        Returns:
          python list of all (sharded) data set files.
        Raises:
          ValueError: if there are not data_files matching the subset.
        """
        pass

    def reader(self):
        """Return a reader for a single entry from the data set.

        See io_ops.py for details of Reader class.

        Returns:
          Reader object that reads the data set.
        """
        return tf.TFRecordReader()

    @abstractmethod
    def preprocessor_type(self):
        pass


class CIFAR10Data(Dataset):
    def __init__(self, subset, data_dir=DATA_PATH):
        super(CIFAR10Data, self).__init__('CIFAR10', subset, data_dir)
        print('using the dataset in ', data_dir)

    def num_classes(self):
        return 10

    def num_examples_per_epoch(self):
        if self.subset == 'train':
            return 50000
        if self.subset == 'validation':
            return 10000

    def data_files(self):
        if self.subset in ['train', 'validation']:
            return [os.path.join(self.data_dir, self.subset + '.tfrecords')]
        else:
            raise ValueError('Invalid data subset "%s"' % self.subset)

    def preprocessor_type(self):
        return CIFAR10Preprocessor