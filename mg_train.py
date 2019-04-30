from datetime import datetime
from tf_utils import *
import os
import time
from tfm_callbacks import CallbackList

def mg_train(model, train_dir, optimizer, layer_to_eqcls,
             max_epochs_per_gpu, max_steps_per_gpu,
             init_step, init_file, load_ckpt, save_final_hdf5,
             ckpt_dir, ckpt_prefix, num_steps_every_ckpt,
             gradient_handler, callbacks=None,
             gpu_idxes=[0], tower_name='tower', bn_layer_to_eqcls=None,
             frequently_save_last_epochs=None,frequently_save_interval=None):

    if max_steps_per_gpu is None:
        max_steps = max_epochs_per_gpu * model.dataset.num_examples_per_epoch() // model.batch_size
        frequently_save_start_steps = (
                                      max_epochs_per_gpu - frequently_save_last_epochs) * model.dataset.num_examples_per_epoch() // model.batch_size
    else:
        assert max_epochs_per_gpu is None
        max_steps = max_steps_per_gpu
        frequently_save_start_steps = max_steps - frequently_save_last_epochs * model.dataset.num_examples_per_epoch() // model.batch_size

    max_steps = int(max_steps)
    print('init step:', init_step)
    print('max training steps per gpu: ', max_steps)
    num_gpus = len(gpu_idxes)
    print('using {} gpus'.format(num_gpus))

    if callbacks is None or len(callbacks) == 0:
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
                        summaries.extend(tf.get_collection(tf.GraphKeys.SUMMARIES, scope))
                        grads = optimizer.compute_gradients(loss)
                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)
                        tower_accs.append(acc_op)
                        if i == 0:
                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = double_bias_gradients(average_gradients(tower_grads))
        grads = gradient_handler.handle_gradient(grads)

        acc = tf.add_n(tower_accs) / num_gpus

        ################################    summary stuff   ############################################
        with tf.name_scope('basics'):
            summaries.append(tf.summary.scalar('average_acc', acc))
            summaries.append(tf.summary.scalar('learning_rate', model.lr))

            kernel_variance_sum = tf.Variable(0., dtype=tf.float32, name='kernel_variance_sum', trainable=False)

            kernel_diff_sum = tf.Variable(0., dtype=tf.float32, name='kernel_diff_sum', trainable=False)
            kernels = model.get_kernel_tensors()
            print(layer_to_eqcls)
            for layer_idx, eqcls in layer_to_eqcls.items():
                kernel_var = kernels[layer_idx]
                if 'depth' in kernel_var.name:
                    for e in eqcls:
                        se = sorted(e)
                        for ei in se[1:]:
                            kernel_diff_sum += tf.reduce_sum(
                                tf.square(kernel_var[:, :, ei, :] - kernel_var[:, :, se[0], :]))
                else:
                    for e in eqcls:
                        if len(e) == 1:
                            continue
                        se = sorted(e)
                        if len(kernel_var.get_shape()) == 4:
                            tmp_sum = 0.0
                            for eee in e:
                                tmp_sum += kernel_var[:, :, :, int(eee)]
                            eee_mean_filter = tmp_sum / len(e)
                            for ei in se[1:]:
                                kernel_diff_sum += tf.reduce_sum(
                                    tf.square(kernel_var[:, :, :, ei] - kernel_var[:, :, :, se[0]]))
                            for ei in se:
                                kernel_variance_sum += tf.reduce_sum(
                                    tf.square(kernel_var[:, :, :, ei] - eee_mean_filter))
                        else:
                            tmp_sum = 0.0
                            for eee in e:
                                tmp_sum += kernel_var[:, int(eee)]
                            eee_mean_filter = tmp_sum / len(e)
                            for ei in se[1:]:
                                kernel_diff_sum += tf.reduce_sum(
                                    tf.square(kernel_var[:, ei] - kernel_var[:, se[0]]))
                            for ei in se:
                                kernel_variance_sum += tf.reduce_sum(
                                    tf.square(kernel_var[:, ei] - eee_mean_filter))


            summaries.append(tf.summary.scalar('diff_kernel', kernel_diff_sum))
            summaries.append(tf.summary.scalar('variance_kernel', kernel_variance_sum))

            moving_variance_diff_sum = tf.Variable(0., dtype=tf.float32, name='moving_variance_diff_sum', trainable=False, )
            gamma_diff_sum = tf.Variable(0., dtype=tf.float32, name='gamma_diff_sum', trainable=False, )
            beta_diff_sum = tf.Variable(0., dtype=tf.float32, name='beta_diff_sum', trainable=False, )

            if bn_layer_to_eqcls is None:
                for layer_idx, eqcls in layer_to_eqcls.items():
                    moving_variance_var = model.get_moving_variance_variable_for_kernel(layer_idx)
                    gamma_var = model.get_gamma_variable_for_kernel(layer_idx)
                    beta_var = model.get_beta_variable_for_kernel(layer_idx)
                    if moving_variance_var is not None:
                        for e in eqcls:
                            se = sorted(e)
                            for ei in se[1:]:
                                moving_variance_diff_sum += tf.reduce_sum(tf.square(moving_variance_var[ei] - moving_variance_var[se[0]]))
                    if gamma_var is not None:
                        for e in eqcls:
                            se = sorted(e)
                            for ei in se[1:]:
                                gamma_diff_sum += tf.reduce_sum(tf.square(gamma_var[ei] - gamma_var[se[0]]))
                    if beta_var is not None:
                        for e in eqcls:
                            se = sorted(e)
                            for ei in se[1:]:
                                beta_diff_sum += tf.reduce_sum(tf.square(beta_var[ei] - beta_var[se[0]]))
            else:
                var_tensors = model.get_moving_variance_tensors()
                gamma_tensors = model.get_gamma_tensors()
                beta_tensors = model.get_beta_tensors()
                for layer_idx, eqcls in bn_layer_to_eqcls.items():
                    moving_variance_var = var_tensors[layer_idx]
                    gamma_var = gamma_tensors[layer_idx]
                    beta_var = beta_tensors[layer_idx]

                    for e in eqcls:
                        se = sorted(e)
                        for ei in se[1:]:
                            moving_variance_diff_sum += tf.reduce_sum(
                                tf.square(moving_variance_var[ei] - moving_variance_var[se[0]]))
                            gamma_diff_sum += tf.reduce_sum(tf.square(gamma_var[ei] - gamma_var[se[0]]))
                            beta_diff_sum += tf.reduce_sum(tf.square(beta_var[ei] - beta_var[se[0]]))

            summaries.append(tf.summary.scalar('diff_gamma', gamma_diff_sum))
            summaries.append(tf.summary.scalar('diff_beta', beta_diff_sum))
            summaries.append(tf.summary.scalar('diff_moving_variance', moving_variance_diff_sum))

        summary_op = tf.summary.merge(summaries)
        ########################################
        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

        # train_op = tf.group(apply_gradient_op, variables_averages_op, *update_ops)
        train_op = tf.group(apply_gradient_op, *update_ops)

        saver = tf.train.Saver(tf.global_variables())

        model.initialize()

        sess = model.sess

        model.load_weights_from_file(init_file)

        # Start the queue runners.
        model.set_and_start_queue_runners()

        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

        if load_ckpt:
            saver.restore(sess, load_ckpt)
            print('restore ckpt from ', load_ckpt)

        if callback_list is not None:
            callback_list.before_train()

        for step in range(init_step, max_steps):

            if callback_list is not None:
                callback_list.before_step(step)

            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = model.batch_size * num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / num_gpus
                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
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

        if save_final_hdf5:
            model.save_weights_to_hdf5(save_final_hdf5)
        if callback_list is not None:
            callback_list.after_train()
        model.stop_queue_runners()







