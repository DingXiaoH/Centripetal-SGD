#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from tensorflow.python.layers.convolutional import conv2d
import numpy as np
from tensorflow.python.layers.core import dense
from tensorflow.python.layers.core import dropout
from tensorflow.python.layers.pooling import max_pooling2d, average_pooling2d
from tensorflow.contrib.layers import flatten, batch_norm
import tensorflow as tf

class ModelBuilder(object):
    def __init__(self, training):
        self.training = training

    def build(self, img_input):
        pass

    def _relu(self, bottom):
        return tf.nn.relu(bottom)

    def _flatten(self, bottom):
        return flatten(bottom)

    def _dropout(self, bottom, drop_rate):
        return dropout(bottom, rate=drop_rate, training=self.training)

    def _batch_norm_default(self, bottom, scope, eps=1e-3, center=True, scale=True):
        if hasattr(self, 'bn_decay'):
            # print('bn decay factor: ', self.bn_decay)
            decay = self.bn_decay
        else:
            decay = 0.9
        if hasattr(self, 'need_gamma'):
            need_gamma = self.need_gamma
        else:
            need_gamma = scale
        if hasattr(self, 'need_beta'):
            need_beta = self.need_beta
        else:
            need_beta = center
        return batch_norm(inputs=bottom, decay=decay, center=need_beta, scale=need_gamma, activation_fn=None,
                   is_training=self.training, scope=scope, epsilon=eps)

    def _maxpool(self, bottom, stride):
        return max_pooling2d(bottom, [stride,stride], [stride,stride])

    def _avgpool(self, bottom, stride):
        return average_pooling2d(bottom, [stride,stride], [stride,stride])

    def _gap(self, bottom):
        height = bottom.get_shape()[1]
        width = bottom.get_shape()[2]
        return average_pooling2d(bottom, [height,width], [height,width])

    def _xavier_initializer(self):
        return tf.contrib.layers.xavier_initializer()




class DC40Builder(ModelBuilder):

    def __init__(self, training, deps):
        super(DC40Builder, self).__init__(training)
        self.deps = deps
        self.N = 12     # 12 layers per stage
        self.num_classes = 10


    def build(self, img_input):
        def conv(input, filters, stride, name):
            return conv2d(input, filters, [3,3], strides=[stride, stride], name=name,
                padding='same', activation=None, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / filters)))

        def add_layer(name, input, cur_layer_idx):
            # shape = input.get_shape().as_list()
            # in_channel = shape[3]
            with tf.variable_scope(name) as scope:
                c = self._batch_norm_default(input, name)
                c = tf.nn.relu(c)
                c = conv(c, self.deps[cur_layer_idx], 1, 'conv1')
                result = tf.concat([input, c], 3)
            return result

        def add_transition(name, input, nb_filters):
            # shape = input.get_shape().as_list()
            # in_channel = shape[3]
            with tf.variable_scope(name) as scope:
                l = self._batch_norm_default(input, name)
                l = tf.nn.relu(l)
                l = conv2d(l, nb_filters, [1,1], strides=[1,1], name='conv1',
                    padding='same', activation=None, use_bias=False)
                l = tf.nn.relu(l)
                l = self._avgpool(l, 2)
            return l

        # tf.summary.image('input-image', img_input)

        l = conv(img_input, self.deps[0], 1, 'conv0')
        with tf.variable_scope('stage1') as scope:
            for i in range(self.N):
                l = add_layer('block{}'.format(i), l, self.N * 0 + 1 + i)
            l = add_transition('transition1', l, nb_filters=self.deps[self.N + 1])

        with tf.variable_scope('stage2') as scope:
            for i in range(self.N):
                l = add_layer('block{}'.format(i), l, self.N * 1 + 2 + i)
            l = add_transition('transition2', l, nb_filters=self.deps[self.N * 2 + 2])

        with tf.variable_scope('stage3') as scope:
            for i in range(self.N):
                l = add_layer('block{}'.format(i), l, self.N * 2 + 3 + i)

        l = self._batch_norm_default(l, scope='bnlast')
        l = tf.nn.relu(l)
        l = self._gap(l)
        l = self._flatten(l)
        logits = dense(l, self.num_classes, activation=None, use_bias=True,
            kernel_initializer=self._xavier_initializer(), name='fc10')

        return logits