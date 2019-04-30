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
"""A library to evaluate Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
from tf_utils import cur_time

def evaluate_model(model, num_examples, results_record_file='tf_test_results.txt', comment='', close_threads=True):
    with model.graph.as_default():

        logits = model.output
        labels = model.input_labels
        sess = model.sess
        loss = model.get_pred_loss(logits, labels, 'none')

        logits_abs_sum = 0.0

        top_1_op = tf.nn.in_top_k(logits, labels, 1)
        top_5_op = tf.nn.in_top_k(logits, labels, 5)

        total_forward_time = 0

        with sess.as_default():

            # Start the queue runners.
            coord = tf.train.Coordinator()
            try:
                threads = []
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                        start=True))

                num_iter = int(math.ceil(num_examples / model.batch_size))
                # Counts the number of correct predictions.
                count_top_1 = 0.0
                count_top_5 = 0.0
                sum_loss = []
                total_sample_count = num_iter * model.batch_size
                step = 0

                print('%s: starting evaluation.' % (datetime.now()))
                start_time = time.time()
                while step < num_iter and not coord.should_stop():
                    one_batch_time = time.time()
                    logits_value, top_1, top_5, loss_value = sess.run([logits, top_1_op, top_5_op, loss])
                    total_forward_time += time.time() - one_batch_time
                    # print(len(pred_class_values))
                    # print(pred_class_values)
                    # label_list.append(label_value)
                    count_top_1 += np.sum(top_1)
                    count_top_5 += np.sum(top_5)
                    sum_loss.append(loss_value)
                    logits_abs_sum += np.sum(np.abs(logits_value))
                    step += 1
                    if step % 20 == 0:
                        duration = time.time() - start_time
                        sec_per_batch = duration / 20.0
                        examples_per_sec = model.batch_size / sec_per_batch
                        print('%s: [%d batches out of %d] (%.1f examples/sec; %.3f'
                              'sec/batch)' % (datetime.now(), step, num_iter,
                                              examples_per_sec, sec_per_batch))
                        start_time = time.time()

                # Compute precision @ 1.
                precision_at_1 = count_top_1 / total_sample_count
                recall_at_5 = count_top_5 / total_sample_count
                loss_vec = np.concatenate(sum_loss)
                mean_loss = np.mean(loss_vec)
                std_loss = np.std(loss_vec)
                print('%s: precision @ 1 = %.4f recall @ 5 = %.4f [%d examples] mean loss = %.8f +- %.8f' %
                      (datetime.now(), precision_at_1, recall_at_5, total_sample_count, mean_loss, std_loss))
                print('logits abs sum: ', logits_abs_sum)

            except Exception as e:
                coord.request_stop(e)

            if close_threads:
                coord.request_stop()
                coord.join(threads, stop_grace_period_secs=10)

    with open(results_record_file, 'a') as f:
        print(cur_time(), model.np_file, precision_at_1, recall_at_5, mean_loss, comment, file=f)
    return total_forward_time, precision_at_1, mean_loss