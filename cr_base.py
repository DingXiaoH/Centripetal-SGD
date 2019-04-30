import os
import numpy as np
import tensorflow as tf
from mg_train import mg_train
from csgd_utils import calculate_eqcls_evenly, calculate_eqcls_biasly, calculate_eqcls_by_kmeans, calculate_bn_eqcls_dc40, tfm_prune_filters_and_save_dc40, tfm_prune_filters_and_save
from tf_utils import log_important
from tfm_callbacks import MergeGradientHandler
from tfm_origin_eval import evaluate_model

CR_OVERALL_LOG_FILE = 'cr_overall_logs.txt'

def cr_base_pipeline(network_type, train_dataset, eval_dataset, train_mode, eval_mode,
                     normal_builder_type, model_type,
                     subsequent_strategy, eqcls_follow_dict,
                     fc_layer_idxes,
                     st_gpu_idxes,
                     l2_factor, eval_batch_size, image_size,

                     try_arg, target_deps, origin_deps, pretrained_model_file,
                     cluster_method, eqcls_layer_idxes,

                     st_batch_size_per_gpu, st_max_epochs_per_gpu, st_lr_epoch_boundaries, st_lr_values,

                     num_steps_per_ckpt_st, frequently_save_interval, frequently_save_last_epochs,

                     diff_factor,

                     schedule_vector,

                     restore_itr=0, restore_st_step=0,

                     init_step=0,

                     slow_on_vec=False
                     ):


    assert cluster_method in ['kmeans', 'even', 'biased']

    origin_deps = np.array(origin_deps)

    prefix = '{}_{}'.format(network_type, try_arg)
    train_dir_pattern = prefix + '_itr{}_train'
    ckpt_dir = prefix + '_ckpt'
    important_log_file = prefix + '_important_log.txt'
    eqcls_file_pattern = prefix + '_itr{}_eqcls.npy'
    sted_weights_pattern = prefix + '_itr{}_sted.hdf5'
    pruned_weights_pattern = prefix + '_itr{}_prunedweights.hdf5'

    target_deps = np.ceil(target_deps).astype(np.int32)
    deps_to_prune = origin_deps - target_deps
    for itr in range(restore_itr, len(schedule_vector)):
        if itr == 0:
            cur_start_model_file = pretrained_model_file
        else:
            cur_start_model_file = pruned_weights_pattern.format(itr - 1)
        if itr == 0:
            cur_remain_deps = np.ceil(origin_deps).astype(np.int32)
        else:
            cur_remain_deps = np.ceil(origin_deps - (deps_to_prune * schedule_vector[itr - 1])).astype(np.int32)

        next_remain_deps = np.ceil(origin_deps - (deps_to_prune * schedule_vector[itr])).astype(np.int32)

        cur_train_dir = train_dir_pattern.format(itr)
        cur_sted_weights = sted_weights_pattern.format(itr)
        cur_pruned_weights = pruned_weights_pattern.format(itr)

        str_start_from = cur_start_model_file or 'SCRATCH'
        log_important(
            'CSGD: start itr {}, start from {}, st to {}, cur remain deps {}, next remain deps {}, cur train dir {}'
                .format(itr, str_start_from, cur_sted_weights,
               list(cur_remain_deps), list(next_remain_deps), cur_train_dir),
            log_file=important_log_file)

        eval_builder = normal_builder_type(training=False, deps=cur_remain_deps)
        train_builder = normal_builder_type(training=True, deps=cur_remain_deps)
        eqcls_file = eqcls_file_pattern.format(itr)

        if itr != restore_itr:
            restore_st_step = 0
        if restore_st_step >= 0:

            # calculate eqcls and test
            test_model = model_type(dataset=eval_dataset, inference_fn=eval_builder.build, mode=eval_mode,
                                 batch_size=eval_batch_size,
                                 image_size=image_size)
            test_model.load_weights_from_file(cur_start_model_file)
            eqcls_dict = {}
            log_important('I calculate eqcls by {}'.format(cluster_method), log_file=important_log_file)
            if os.path.exists(eqcls_file):
                eqcls_dict = np.load(eqcls_file).item()

            for i in eqcls_layer_idxes:
                if i in eqcls_dict:
                    continue
                if cluster_method == 'kmeans':
                    eqcls = calculate_eqcls_by_kmeans(test_model, i, next_remain_deps[i])
                elif cluster_method == 'even':
                    eqcls = calculate_eqcls_evenly(filters=cur_remain_deps[i], num_eqcls=next_remain_deps[i])
                elif cluster_method == 'biased':
                    eqcls = calculate_eqcls_biasly(filters=cur_remain_deps[i], num_eqcls=next_remain_deps[i])
                else:
                    assert False
                eqcls_dict[i] = eqcls
                np.save(eqcls_file, eqcls_dict)

            if eqcls_follow_dict is not None:
                for k, v in eqcls_follow_dict.items():
                    if v in eqcls_dict:
                        eqcls_dict[k] = eqcls_dict[v]

            if network_type == 'dc40':
                bn_layer_to_eqcls = calculate_bn_eqcls_dc40(eqcls_dict)
            else:
                bn_layer_to_eqcls = None

            #   test 1
            evaluate_model(test_model, num_examples=eval_dataset.num_examples_per_epoch(),
                           results_record_file=important_log_file,
                           comment='eval at itr {}, cur remain deps {}'.format(itr, cur_remain_deps))
            del test_model

            if st_max_epochs_per_gpu <= 0:  # skip C-SGD training
                cur_sted_weights = cur_start_model_file
            else:
                #   C-SGD train
                #   Note that l2_factor = 0 (cancel the original L2 regularization term implemented by base_loss += 0.5 * l2_factor * sum(kernels ** 2)
                train_model = model_type(dataset=train_dataset, inference_fn=train_builder.build, mode=train_mode,
                                         batch_size=st_batch_size_per_gpu, image_size=image_size, l2_factor=0,
                                         deps=cur_remain_deps)
                gradient_handler = MergeGradientHandler(model=train_model, layer_to_eqcls=eqcls_dict,
                                                        l2_factor=l2_factor,
                                                        diff_factor=diff_factor, exclude_l2_decay_keywords=None,
                                                        bn_layer_to_eqcls=bn_layer_to_eqcls,
                                                        version=2, slow_on_vec=slow_on_vec)
                cur_init_step = init_step if itr == restore_itr else 0
                lr = train_model.get_piecewise_lr(st_lr_values, boundaries_epochs=st_lr_epoch_boundaries, init_step=cur_init_step)
                optimizer = tf.train.MomentumOptimizer(lr, momentum=0.9, use_nesterov=True)
                if itr == restore_itr and restore_st_step > 0:
                    load_ckpt = os.path.join(cur_train_dir, 'model.ckpt-{}'.format(restore_st_step))
                    print('the CSGD train restarts from ', load_ckpt)
                else:
                    load_ckpt = None
                if itr == restore_itr:
                    start_step = max(init_step, restore_st_step)
                else:
                    start_step = 0
                mg_train(model=train_model, train_dir=cur_train_dir, optimizer=optimizer, layer_to_eqcls=eqcls_dict,
                         max_epochs_per_gpu=st_max_epochs_per_gpu, max_steps_per_gpu=None,
                         init_step=start_step, init_file=cur_start_model_file, load_ckpt=load_ckpt,
                         save_final_hdf5=cur_sted_weights,
                         ckpt_dir=ckpt_dir, ckpt_prefix='st_ckpt_itr{}'.format(itr),
                         num_steps_every_ckpt=num_steps_per_ckpt_st,
                         gradient_handler=gradient_handler, gpu_idxes=st_gpu_idxes, bn_layer_to_eqcls=bn_layer_to_eqcls,
                        frequently_save_interval=frequently_save_interval, frequently_save_last_epochs=frequently_save_last_epochs)
                log_important('C-SGD train completed. save to: {}'.format(cur_sted_weights),
                              log_file=important_log_file)
                del train_model

        else:
            print('load eqcls form file: ', eqcls_file)
            eqcls_dict = np.load(eqcls_file).item()
            if eqcls_follow_dict is not None:
                for k, v in eqcls_follow_dict.items():
                    eqcls_dict[k] = eqcls_dict[v]

        #   test after CSGD train before trimming
        test_model = model_type(dataset=eval_dataset, inference_fn=eval_builder.build, mode=eval_mode,
            batch_size=eval_batch_size, image_size=image_size)
        test_model.load_weights_from_file(cur_sted_weights)
        evaluate_model(test_model, num_examples=eval_dataset.num_examples_per_epoch(),
            results_record_file=important_log_file,
            comment='eval before trimming at itr {}, cur remain deps {}'.format(itr, cur_remain_deps),
            close_threads=False)
        del test_model

        # Trim the C-SGD trained weights
        trim_model = model_type(dataset=eval_dataset, inference_fn=eval_builder.build, mode=eval_mode,
            batch_size=eval_batch_size, image_size=image_size)
        trim_model.load_weights_from_file(cur_sted_weights)
        if network_type == 'dc40':
            bn_layer_to_eqcls = calculate_bn_eqcls_dc40(eqcls_dict)
            tfm_prune_filters_and_save_dc40(trim_model, eqcls_dict, bn_layer_to_eqcls=bn_layer_to_eqcls,
                save_file=cur_pruned_weights,
                new_deps=next_remain_deps)
        else:
            tfm_prune_filters_and_save(trim_model, eqcls_dict, save_file=cur_pruned_weights,
                fc_layer_idxes=fc_layer_idxes, subsequent_strategy=subsequent_strategy, new_deps=next_remain_deps)

        log_important('finished trimming at itr {}, save to {}'.format(itr, cur_pruned_weights),
            log_file=important_log_file)
        del trim_model

        #   test the trimmed model
        pruned_builder = normal_builder_type(training=False, deps=next_remain_deps)
        pruned_test_model = model_type(dataset=eval_dataset, inference_fn=pruned_builder.build, mode=eval_mode,
            batch_size=eval_batch_size, image_size=image_size)
        pruned_test_model.load_weights_from_file(cur_pruned_weights)
        evaluate_model(pruned_test_model, num_examples=eval_dataset.num_examples_per_epoch(),
            results_record_file=important_log_file,
            comment='eval after trimming at itr {}, cur remain deps {}'.format(itr, next_remain_deps))
        del pruned_test_model