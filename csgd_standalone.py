from tfm_model import TFModel
from tf_dataset import CIFAR10Data
from tfm_origin_train import train
from tfm_builder_densenet import DC40Builder
from tfm_constants import DC40_ORIGIN_DEPS, customized_dc40_deps, DC40_ALL_CONV_LAYERS
import tensorflow as tf
import sys
from cr_dc40 import cr_dc40
import os
import numpy as np
from csgd_utils import calculate_bn_eqcls_dc40, tfm_prune_filters_and_save_dc40
from tfm_origin_eval import evaluate_model
from tf_utils import extract_deps_from_weights_file

PRETRAINED_MODEL_FILE = 'std_dc40_9382.hdf5'


LR_VALUES = [3e-3, 3e-4, 3e-5, 3e-6]
LR_BOUNDARIES = [200, 400, 500]
MAX_EPOCHS = 600
BATCH_SIZE = 64

EPSILON_VALUE = 3e-3

TARGET_DEPS = customized_dc40_deps('3-3-3')     # 3 filters per incremental conv layer (the original is 12)

DC40_L2_FACTOR = 1e-4



def eval_model(weights_path):
    dataset = CIFAR10Data('validation')
    deps = extract_deps_from_weights_file(weights_path)
    if deps is None:
        deps = DC40_ORIGIN_DEPS
    builder = DC40Builder(training=False, deps=deps)
    model = TFModel(dataset, builder.build, 'eval', batch_size=250, image_size=32)
    model.load_weights_from_file(weights_path)
    evaluate_model(model, num_examples=dataset.num_examples_per_epoch(), results_record_file='origin_dc40_eval_record.txt')


def compare_csgd(prefix):
    cr_dc40(prefix, target_deps=TARGET_DEPS, origin_deps=DC40_ORIGIN_DEPS, pretrained_model_file=PRETRAINED_MODEL_FILE,
        cluster_method='kmeans', eqcls_layer_idxes=DC40_ALL_CONV_LAYERS,
        st_batch_size_per_gpu=BATCH_SIZE, st_max_epochs_per_gpu=MAX_EPOCHS, st_lr_epoch_boundaries=LR_BOUNDARIES,
        st_lr_values=LR_VALUES,
        diff_factor=EPSILON_VALUE, schedule_vector=[1], slow_on_vec=False)


#   In order to reuse the pruning function, we produce $\mathcal{C}$ based on the magnitude of filter kernels
#   thus pruning according to it becomes equivalent to pruning the 9 filters smaller in magnitude at each layer
def _produce_magnitude_equivalent_eqcls(target_deps, save_path):
    builder = DC40Builder(True, deps=DC40_ORIGIN_DEPS)
    prune_model = TFModel(CIFAR10Data('train'), builder.build, 'train', batch_size=BATCH_SIZE, image_size=32, l2_factor=DC40_L2_FACTOR, deps=DC40_ORIGIN_DEPS)
    prune_model.load_weights_from_file(PRETRAINED_MODEL_FILE)

    equivalent_dict_eqcls = {}
    for i in DC40_ALL_CONV_LAYERS:
        kernel_value = prune_model.get_value(prune_model.get_kernel_tensors()[i])
        summed_kernel_value = np.sum(np.abs(kernel_value), axis=(0, 1, 2))
        assert len(summed_kernel_value) == DC40_ORIGIN_DEPS[i]
        index_array = np.argsort(summed_kernel_value)
        index_to_delete = index_array[:(DC40_ORIGIN_DEPS[i] - target_deps[i])]
        cur_eqcls = []
        for k in range(DC40_ORIGIN_DEPS[i]):
            if k not in index_to_delete:
                cur_eqcls.append([k])
        for k in index_to_delete:
            cur_eqcls[0].append(k)
        equivalent_dict_eqcls[i] = cur_eqcls

    np.save(save_path, equivalent_dict_eqcls)
    del prune_model
    return equivalent_dict_eqcls


def compare_magnitude(prefix):
    pruned = 'dc40_{}_prunedweights.hdf5'.format(prefix)
    target_deps = TARGET_DEPS
    save_hdf5 = '{}_trained.hdf5'.format(prefix)
    equivalent_eqcls_path = 'dc40_equivalent_eqcls_{}.npy'.format(prefix)
    if not os.path.exists(pruned):
        eqcls_dict = _produce_magnitude_equivalent_eqcls(target_deps=target_deps, save_path=equivalent_eqcls_path)
        bn_layer_to_eqcls = calculate_bn_eqcls_dc40(eqcls_dict)

        builder = DC40Builder(False, deps=DC40_ORIGIN_DEPS)
        prune_model = TFModel(CIFAR10Data('train'), builder.build, 'eval', batch_size=64, image_size=32,
            l2_factor=1e-4, deps=DC40_ORIGIN_DEPS)
        prune_model.load_weights_from_file(PRETRAINED_MODEL_FILE)
        tfm_prune_filters_and_save_dc40(prune_model, eqcls_dict, bn_layer_to_eqcls=bn_layer_to_eqcls,
            save_file=pruned, new_deps=target_deps)
        del prune_model

    builder = DC40Builder(True, deps=target_deps)
    model = TFModel(CIFAR10Data('train'), builder.build, 'eval', batch_size=BATCH_SIZE, image_size=32,  #TODO eval?
        l2_factor=DC40_L2_FACTOR, deps=target_deps)
    lr = model.get_piecewise_lr(values=LR_VALUES, boundaries_epochs=LR_BOUNDARIES)
    optimizer = tf.train.MomentumOptimizer(lr, momentum=0.9, use_nesterov=True)
    train(model, train_dir='{}_train'.format(prefix), optimizer=optimizer, max_epochs_per_gpu=MAX_EPOCHS,
        gpu_idxes=[0], init_file=pruned, save_final_hdf5=save_hdf5,
        ckpt_dir='{}_ckpt'.format(prefix), ckpt_prefix=prefix, num_steps_every_ckpt=20000)



if __name__ == '__main__':
    prefix = sys.argv[1]
    if 'csgd' in prefix:
        compare_csgd(prefix)
    elif 'magnitude' in prefix:
        compare_magnitude(prefix)
    elif prefix == 'eval':
        eval_model(sys.argv[2])
    else:
        assert False
