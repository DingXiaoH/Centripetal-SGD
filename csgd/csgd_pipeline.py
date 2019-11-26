from ding_train import ding_train
import os
from ding_test import general_test
from csgd.csgd_train import csgd_train_and_prune

def csgd_prune_pipeline(init_hdf5, base_train_config, csgd_train_config,
                        target_deps, centri_strength, pacesetter_dict, succeeding_strategy):
    #   If there is no given base weights file, train from scratch.
    if init_hdf5 is None:
        csgd_init_weights = os.path.join(base_train_config.output_dir, 'finish.hdf5')
        if not os.path.exists(csgd_init_weights):
            ding_train(cfg=base_train_config)
    else:
        csgd_init_weights = init_hdf5

    #   C-SGD train then prune
    pruned_weights = os.path.join(csgd_train_config.output_dir, 'pruned.hdf5')
    csgd_train_and_prune(cfg=csgd_train_config,
                        target_deps=target_deps, centri_strength=centri_strength,
                         pacesetter_dict=pacesetter_dict, succeeding_strategy=succeeding_strategy,
                         pruned_weights=pruned_weights,
                         init_hdf5=csgd_init_weights)  # TODO init?

    #   Test it.
    general_test(csgd_train_config.network_type, weights=pruned_weights)


def csgd_iterative(init_hdf5, base_train_config, csgd_train_config,
                        itr_deps, centri_strength, pacesetter_dict, succeeding_strategy):
    for itr, deps in enumerate(itr_deps):
        if itr == 0:
            begin_weights = init_hdf5
        else:
            begin_weights = os.path.join(csgd_train_config.output_dir, 'itr{}'.format(itr-1), 'pruned.hdf5')
        itr_output_dir = os.path.join(csgd_train_config.output_dir, 'itr{}'.format(itr))
        if os.path.exists(os.path.join(itr_output_dir, 'pruned.hdf5')):
            continue
        itr_csgd_config = csgd_train_config._replace(tb_dir=itr_output_dir)._replace(output_dir=itr_output_dir)
        if itr != 0:
            itr_csgd_config = itr_csgd_config._replace(deps=itr_deps[itr-1])
        csgd_prune_pipeline(init_hdf5=begin_weights, base_train_config=base_train_config,
                            csgd_train_config=itr_csgd_config, target_deps=deps,
                            centri_strength=centri_strength, pacesetter_dict=pacesetter_dict,
                            succeeding_strategy=succeeding_strategy)

