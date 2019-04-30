from cr_base import cr_base_pipeline
from tf_dataset import CIFAR10Data
from tfm_model import TFModel
from tfm_constants import *
from tfm_builder_densenet import DC40Builder

def cr_dc40(
        try_arg, target_deps, origin_deps, pretrained_model_file,
        cluster_method, eqcls_layer_idxes,

        st_batch_size_per_gpu, st_max_epochs_per_gpu, st_lr_epoch_boundaries, st_lr_values,

        diff_factor,
        schedule_vector,

        restore_itr=0, restore_st_step=0,
        init_step=0,
        frequently_save_interval=None, frequently_save_last_epochs=None,
        slow_on_vec=False
):
    cr_base_pipeline(
        network_type='dc40',
        train_dataset=CIFAR10Data('train'), eval_dataset=CIFAR10Data('validation'),
        train_mode='train', eval_mode='eval',
        normal_builder_type=DC40Builder, model_type=TFModel,
        subsequent_strategy=DC40_SUBSEQUENT_STRATEGY, eqcls_follow_dict=DC40_FOLLOW_DICT,
        fc_layer_idxes=DC40_FC_LAYERS, st_gpu_idxes=[0],
        l2_factor=1e-4, eval_batch_size=500, image_size=32,
        frequently_save_interval=frequently_save_interval or 1000, frequently_save_last_epochs=frequently_save_last_epochs or 50, num_steps_per_ckpt_st=20000,

        try_arg=try_arg, target_deps=target_deps, origin_deps=origin_deps, pretrained_model_file=pretrained_model_file,
        cluster_method=cluster_method, eqcls_layer_idxes=eqcls_layer_idxes,
        st_batch_size_per_gpu=st_batch_size_per_gpu, st_max_epochs_per_gpu=st_max_epochs_per_gpu, st_lr_epoch_boundaries=st_lr_epoch_boundaries, st_lr_values=st_lr_values,
        diff_factor=diff_factor,
        schedule_vector=schedule_vector,
        restore_itr=restore_itr, restore_st_step=restore_st_step,
        init_step=init_step,
        slow_on_vec=slow_on_vec
    )