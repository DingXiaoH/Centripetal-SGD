import tensorflow as tf
import numpy as np
import re
import h5py
import time
import os

def latest_checkpoint_abs_path(train_dir):
    ckpt_file = os.path.join(train_dir, 'checkpoint')
    with open(ckpt_file, 'r') as f:
        first_line = f.readline().strip()
    ckpt_name = first_line.split(' ')[1].replace('"','')
    return os.path.join(train_dir, ckpt_name)


def extract_deps_from_weights_file(file_path):
    if file_path.endswith('npy'):
        weight_dic = np.load(file_path).item()
    else:
        weight_dic = read_hdf5(file_path)
    if 'deps' in weight_dic:
        return weight_dic['deps']
    else:
        return None


def cur_time():
    return time.strftime('%Y,%b,%d,%X')


def representsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def save_hdf5(numpy_dict, file_path):
    with h5py.File(file_path, 'w') as f:
        for k,v in numpy_dict.items():
            f.create_dataset(str(k).replace('/','+'), data=v)
    print('saved {} arrays to {}'.format(len(numpy_dict), file_path))

def read_hdf5(file_path):
    result = {}
    with h5py.File(file_path, 'r') as f:
        for k in f.keys():
            value = np.asarray(f[k])
            if representsInt(k):
                result[int(k)] = value
            else:
                result[str(k).replace('+','/')] = value
    print('read {} arrays from {}'.format(len(result), file_path))
    return result


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def log_important(message, log_file):
    print(message)
    with open(log_file, 'a') as f:
        print(message, file=f)


def eliminate_all_patterns(text, patterns):
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    return text


def double_bias_gradients(origin_gradients):
    bias_cnt = 0
    result = []
    print('doubling bias gradients')
    for grad, var in origin_gradients:
        if 'bias' in var.name:
            result.append((2 * grad, var))
            bias_cnt += 1
        else:
            result.append((grad, var))
    print('doubled gradients for {} bias variables'.format(bias_cnt))
    return result


def tf_get_gradient_by_var(grads_and_vars, var):
    for (g, v) in grads_and_vars:
        if v.name == var.name:
            return g
    return None