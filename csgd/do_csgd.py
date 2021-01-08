from constants import *
from base_config import get_baseconfig_by_epoch
from model_map import get_dataset_name_by_model_name
from csgd.flops_scripts import *
import argparse
from csgd.csgd_pipeline import csgd_iterative


thinet_schedule_vector = [0.7, 0.5, 0.3]

schedule_vector_7060504030 = [0.7, 0.6, 0.5, 0.4, 0.3]

def generate_itr_to_target_deps_by_schedule_vector(schedule_vec):
    result = []
    for m in schedule_vec:
        d = np.array(RESNET50_ORIGIN_DEPS_FLATTENED)
        for i in RESNET50_INTERNAL_KERNEL_IDXES:
            d[i] = np.ceil(d[i] * m).astype(np.int32)
        result.append(d)
    return result

default_itr_to_target_deps = generate_itr_to_target_deps_by_schedule_vector(thinet_schedule_vector)

itr_target_deps_7060504030 = generate_itr_to_target_deps_by_schedule_vector(schedule_vector_7060504030)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--arch', default='src56')
    parser.add_argument('-c', '--conti_or_fs', default='fs')
    parser.add_argument('-i', '--begin_itr', default=0)
    parser.add_argument(
        '--local_rank', default=0, type=int,
        help='process rank on node')

    start_arg = parser.parse_args()

    network_type = start_arg.arch
    conti_or_fs = start_arg.conti_or_fs
    assert conti_or_fs in ['continue', 'fs']
    auto_continue = conti_or_fs == 'continue'
    print('auto continue: ', auto_continue)

    if network_type == 'sres50':
        weight_decay_strength = 1e-4
        batch_size = 256
        deps = RESNET50_ORIGIN_DEPS_FLATTENED
        succeeding_strategy = resnet_bottleneck_succeeding_strategy(50)
        pacesetter_dict = resnet_bottleneck_follow_dict(50)
        init_hdf5 = 'torchvision_res50.hdf5'
        flops_func = calculate_resB_50_flops
        train_base_config = None

        warmup_epochs = 5
        lrs = LRSchedule(base_lr=3e-3, max_epochs=70, lr_epoch_boundaries=[30, 50, 60], lr_decay_factor=0.1,
                         linear_final_lr=None, cosine_minimum=None)
        itr_deps = itr_target_deps_7060504030
        centri_strength = 0.05


    elif network_type == 'src56':
        weight_decay_strength = 1e-4
        batch_size = 64
        deps = rc_origin_deps_flattened(9)
        succeeding_strategy = rc_succeeding_strategy(9)
        pacesetter_dict = rc_pacesetter_dict(9)
        base_log_dir = 'src56_train'
        init_hdf5 = None
        flops_func = calculate_rc56_flops
        base_lrs = LRSchedule(base_lr=5e-2, max_epochs=600, lr_epoch_boundaries=[200, 400], lr_decay_factor=0.1,
                              linear_final_lr=None, cosine_minimum=None)
        train_base_config = get_baseconfig_by_epoch(network_type=network_type, dataset_name=get_dataset_name_by_model_name(network_type), dataset_subset='train',
                                     global_batch_size=batch_size, num_node=1,
                                     weight_decay=weight_decay_strength, optimizer_type='sgd', momentum=0.9,
                                     max_epochs=base_lrs.max_epochs, base_lr=base_lrs.base_lr, lr_epoch_boundaries=base_lrs.lr_epoch_boundaries,
                                     lr_decay_factor=base_lrs.lr_decay_factor, cosine_minimum=base_lrs.cosine_minimum,
                                     warmup_epochs=5, warmup_method='linear', warmup_factor=0,
                                     ckpt_iter_period=20000, tb_iter_period=100, output_dir=base_log_dir,
                                     tb_dir=base_log_dir, save_weights=None, val_epoch_period=2, linear_final_lr=base_lrs.linear_final_lr,
                                     weight_decay_bias=0, deps=deps)

        warmup_epochs = 0

        lrs = LRSchedule(base_lr=3e-2, max_epochs=600, lr_epoch_boundaries=[200, 400], lr_decay_factor=0.1,
                         linear_final_lr=None, cosine_minimum=None)
        itr_deps = [[d * 13 // 16 for d in deps],
                         [d * 11 // 16 for d in deps],
                         [d * 5 // 8 for d in deps]]
        centri_strength = 3e-3

    else:
        raise ValueError('...')

    log_dir = 'csgd_models/{}_train'.format(network_type)

    weight_decay_bias = 0
    warmup_factor = 0

    csgd_config = get_baseconfig_by_epoch(network_type=network_type,
                                     dataset_name=get_dataset_name_by_model_name(network_type), dataset_subset='train',
                                     global_batch_size=batch_size, num_node=1,
                                     weight_decay=weight_decay_strength, optimizer_type='sgd', momentum=0.9,
                                     max_epochs=lrs.max_epochs, base_lr=lrs.base_lr, lr_epoch_boundaries=lrs.lr_epoch_boundaries, cosine_minimum=lrs.cosine_minimum,
                                     lr_decay_factor=lrs.lr_decay_factor,
                                     warmup_epochs=warmup_epochs, warmup_method='linear', warmup_factor=warmup_factor,
                                     ckpt_iter_period=40000, tb_iter_period=100, output_dir=log_dir,
                                     tb_dir=log_dir, save_weights=None, val_epoch_period=5, linear_final_lr=lrs.linear_final_lr,
                                     weight_decay_bias=weight_decay_bias, deps=deps)

    if start_arg.local_rank == 0:
        print('=====================================')
        print('prune {}, the original deps is {}, flops is {}'.format(network_type, deps, flops_func(deps)))
        for itr, cur_deps in enumerate(itr_deps):
            print('itr {}, the target deps is {}, flops is {}'.format(itr, cur_deps, flops_func(cur_deps)))
        print('=====================================')

    csgd_iterative(local_rank=start_arg.local_rank, init_hdf5=init_hdf5, base_train_config=train_base_config,
                   csgd_train_config=csgd_config, itr_deps=itr_deps, centri_strength=centri_strength,
                   pacesetter_dict=pacesetter_dict, succeeding_strategy=succeeding_strategy, begin_itr=int(start_arg.begin_itr))