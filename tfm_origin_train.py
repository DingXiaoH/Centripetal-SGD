from datetime import datetime
from tf_utils import *
import os
import time
from tfm_callbacks import CallbackList

def train(model, train_dir, optimizer, max_epochs_per_gpu=None, max_steps_per_gpu=None, gpu_idxes=[0],
          tower_name='tower', moving_average_decay=0.9999, init_file=None,
          load_ckpt=None, save_final_np=None, save_final_hdf5=None, save_final_mvav_hdf5=None,
          ckpt_dir = None, ckpt_prefix=None, num_steps_every_ckpt=5000, init_step=0,
          callbacks=None, gradient_handler=None, histogram_keywords=None,
          frequently_save_last_epochs=0, frequently_save_interval=None):

    if histogram_keywords == 'common':
        histogram_keywords = ['kernel', 'bias', 'moving_mean', 'moving_variance', 'gamma', 'beta', 'lmd']
    elif histogram_keywords is None:
        histogram_keywords = []

    if load_ckpt == 'auto':
        if init_step is not None:
            load_ckpt = os.path.join(train_dir, 'model.ckpt-{}'.format(init_step))
        else:
            load_ckpt = latest_checkpoint_abs_path(train_dir)


    if max_steps_per_gpu is None:
        max_steps = max_epochs_per_gpu * model.dataset.num_examples_per_epoch() // model.batch_size
        frequently_save_start_steps = (
                                      max_epochs_per_gpu - frequently_save_last_epochs) * model.dataset.num_examples_per_epoch() // model.batch_size
    else:
        assert max_epochs_per_gpu is None
        max_steps = max_steps_per_gpu
        frequently_save_start_steps = max_steps - frequently_save_last_epochs * model.dataset.num_examples_per_epoch() // model.batch_size

    max_steps = int(max_steps)
    print('max training steps per gpu: ', max_steps)
    num_gpus = len(gpu_idxes)
    print('using {} gpus'.format(num_gpus))

    if callbacks is None or (type(callbacks) is list and len(callbacks) == 0):
        callback_list = None
    else:
        callback_list = CallbackList(callbacks)

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    print('train dir: ', train_dir)

    if ckpt_dir and not os.path.isdir(ckpt_dir):
        os.mkdir(ckpt_dir)

    with model.graph.as_default(), tf.device('/cpu:0'):
        global_step = model.global_step
        images = model.input_images
        labels = model.input_labels

        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
            [images, labels], capacity=2 * num_gpus)
        # Calculate the gradients for each model tower.
        tower_grads = []
        tower_accs = []
        summaries = []

        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.device('/gpu:%d' % gpu_idxes[i]):
                    with tf.name_scope('%s_%d' % (tower_name, gpu_idxes[i])) as scope:
                        # Dequeues one batch for the GPU
                        image_batch, label_batch = batch_queue.dequeue()
                        # "reuse" function is called inside this method
                        loss, acc_op = model.get_tower_loss_and_acc(scope, image_batch, label_batch, tower_name=tower_name)
                        # Retain the summaries from the final tower.
                        summaries.extend(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope))
                        grads = optimizer.compute_gradients(loss)
                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)
                        tower_accs.append(acc_op)
                        if i == 0:
                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

        grads = double_bias_gradients(average_gradients(tower_grads))
        if hasattr(model, 'modify_gradient'):
            grads = model.modify_gradient(grads)
        if gradient_handler is not None:
            grads = gradient_handler.handle_gradient(grads)

        acc = tf.add_n(tower_accs) / num_gpus

        with tf.name_scope('basics'):
            summaries.append(tf.summary.scalar('average_acc', acc))
            summaries.append(tf.summary.scalar('learning_rate', model.lr))


        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        # # Add histograms for key variables.
        key_vars = model.get_key_variables()
        for var in key_vars:
            need_his = False
            for k in histogram_keywords:
                if k in var.name:
                    need_his = True
                    break
            if need_his:
                summaries.append(tf.summary.histogram(var.op.name, var))
            else:
                print('need no histogram: ', var.name)

        # Track the moving averages of all trainable variables.
        if save_final_mvav_hdf5:
            variable_averages = tf.train.ExponentialMovingAverage(
                moving_average_decay, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            # Group all updates to into a single train op.
            train_op = tf.group(apply_gradient_op, variables_averages_op, *update_ops)
        else:
            train_op = tf.group(apply_gradient_op, *update_ops)


        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)   #TODO
        summary_op = tf.summary.merge(summaries)

        model.initialize()

        sess = model.sess

        if init_file is not None:
            model.load_weights_from_hdf5(init_file)
        else:
            print('********************* train from scratch *******************')

        # Start the queue runners.
        model.set_and_start_queue_runners()

        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        if load_ckpt:
            saver.restore(sess, load_ckpt)

        if callback_list is not None:
            callback_list.before_train()

        for step in range(init_step, max_steps):

            if callback_list is not None:
                callback_list.before_step(step)

            start_time = time.time()
            # imgs, labels = sess.run([image_batch, label_batch])
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = model.batch_size * num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / num_gpus
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
            # Save the model checkpoint periodically.
            if step % 2000 == 0 or (step + 1) == max_steps:
                checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            if step + 1 > frequently_save_start_steps:
                if (step+1) % frequently_save_interval == 0:
                    save_path = os.path.join(ckpt_dir, '{}_step_{}'.format(ckpt_prefix, step + 1))
                    model.save_weights_to_hdf5(save_path)
            else:
                if num_steps_every_ckpt > 0 and (step + 1) % num_steps_every_ckpt == 0:
                    save_path = os.path.join(ckpt_dir, '{}_step_{}'.format(ckpt_prefix, step + 1))
                    model.save_weights_to_hdf5(save_path)

            if callback_list is not None:
                callback_list.after_step(step)

        if save_final_np:
            model.save_weights_to_np(save_final_np)
        if save_final_hdf5:
            model.save_weights_to_hdf5(save_final_hdf5)
        if save_final_mvav_hdf5:
            model.save_moving_average_weights_to_hdf5(save_final_mvav_hdf5, variable_averages)
        if callback_list is not None:
            callback_list.after_train()

        model.stop_queue_runners()
        sess.close()