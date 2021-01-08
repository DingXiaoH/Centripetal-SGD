from torch.utils.tensorboard import SummaryWriter
from base_config import BaseConfigByEpoch
from model_map import get_model_fn
from data.data_factory import create_dataset, load_cuda_data, num_iters_per_epoch
from torch.nn.modules.loss import CrossEntropyLoss
from utils.engine import Engine
from utils.pyt_utils import ensure_dir
from utils.misc import torch_accuracy, AvgMeter
from collections import OrderedDict
import torch
from tqdm import tqdm
import time
from builder import ConvBuilder
from utils.lr_scheduler import get_lr_scheduler
import os
from utils.checkpoint import get_last_checkpoint
from ndp_test import val_during_train
from sklearn.cluster import KMeans
import numpy as np
from csgd.csgd_prune import csgd_prune_and_save

TRAIN_SPEED_START = 0.1
TRAIN_SPEED_END = 0.2

COLLECT_TRAIN_LOSS_EPOCHS = 3

TEST_BATCH_SIZE = 100

KERNEL_KEYWORD = 'conv.weight'

def add_vecs_to_merge_mat_dicts(param_name_to_merge_matrix):
    kernel_names = set(param_name_to_merge_matrix.keys())
    for name in kernel_names:
        bias_name = name.replace(KERNEL_KEYWORD, 'conv.bias')
        gamma_name = name.replace(KERNEL_KEYWORD, 'bn.weight')
        beta_name = name.replace(KERNEL_KEYWORD, 'bn.bias')
        param_name_to_merge_matrix[bias_name] = param_name_to_merge_matrix[name]
        param_name_to_merge_matrix[gamma_name] = param_name_to_merge_matrix[name]
        param_name_to_merge_matrix[beta_name] = param_name_to_merge_matrix[name]

def generate_merge_matrix_for_kernel(deps, layer_idx_to_clusters, kernel_namedvalue_list):
    result = {}
    for layer_idx, clusters in layer_idx_to_clusters.items():
        num_filters = deps[layer_idx]
        merge_trans_mat = np.zeros((num_filters, num_filters), dtype=np.float32)
        for clst in clusters:
            if len(clst) == 1:
                merge_trans_mat[clst[0], clst[0]] = 1
                continue
            sc = sorted(clst)       # Ideally, clst should have already been sorted in ascending order
            for ei in sc:
                for ej in sc:
                    merge_trans_mat[ei, ej] = 1 / len(clst)
        result[kernel_namedvalue_list[layer_idx].name] = torch.from_numpy(merge_trans_mat).cuda()
    return result


#   Recently it is popular to cancel weight decay on vecs
def generate_decay_matrix_for_kernel_and_vecs(deps, layer_idx_to_clusters, kernel_namedvalue_list,
                                              weight_decay, weight_decay_bias, centri_strength):
    result = {}
    #   for the kernel
    for layer_idx, clusters in layer_idx_to_clusters.items():
        num_filters = deps[layer_idx]
        decay_trans_mat = np.zeros((num_filters, num_filters), dtype=np.float32)
        for clst in clusters:
            sc = sorted(clst)
            for ee in sc:
                decay_trans_mat[ee, ee] = weight_decay + centri_strength
                for p in sc:
                    decay_trans_mat[ee, p] += -centri_strength / len(clst)
        kernel_mat = torch.from_numpy(decay_trans_mat).cuda()
        result[kernel_namedvalue_list[layer_idx].name] = kernel_mat

    #   for the vec params (bias, beta and gamma), we use 0.1 * centripetal strength
    for layer_idx, clusters in layer_idx_to_clusters.items():
        num_filters = deps[layer_idx]
        decay_trans_mat = np.zeros((num_filters, num_filters), dtype=np.float32)
        for clst in clusters:
            sc = sorted(clst)
            for ee in sc:
                # Note: using smaller centripetal strength on the scaling factor of BN improve the performance in some of the cases
                decay_trans_mat[ee, ee] = weight_decay_bias + centri_strength * 0.1
                for p in sc:
                    decay_trans_mat[ee, p] += -centri_strength * 0.1 / len(clst)
        vec_mat = torch.from_numpy(decay_trans_mat).cuda()
        result[kernel_namedvalue_list[layer_idx].name.replace(KERNEL_KEYWORD, 'bn.weight')] = vec_mat
        result[kernel_namedvalue_list[layer_idx].name.replace(KERNEL_KEYWORD, 'bn.bias')] = vec_mat
        result[kernel_namedvalue_list[layer_idx].name.replace(KERNEL_KEYWORD, 'conv.bias')] = vec_mat

    return result

def cluster_by_kmeans(kernel_value, num_cluster):
    assert kernel_value.ndim == 4
    x = np.reshape(kernel_value, (kernel_value.shape[0], -1))
    if num_cluster == x.shape[0]:
        result = [[i] for i in range(num_cluster)]
        return result
    else:
        print('cluster {} filters into {} clusters'.format(x.shape[0], num_cluster))
    km = KMeans(n_clusters=num_cluster)
    km.fit(x)
    result = []
    for j in range(num_cluster):
        result.append([])
    for i, c in enumerate(km.labels_):
        result[c].append(i)
    for r in result:
        assert len(r) > 0
    return result

def _is_follower(layer_idx, pacesetter_dict):
    followers_and_pacesetters = set(pacesetter_dict.keys())
    return (layer_idx in followers_and_pacesetters) and (pacesetter_dict[layer_idx] != layer_idx)

def get_layer_idx_to_clusters(kernel_namedvalue_list, target_deps, pacesetter_dict):
    result = {}
    for layer_idx, named_kv in enumerate(kernel_namedvalue_list):
        num_filters = named_kv.value.shape[0]
        if pacesetter_dict is not None and _is_follower(layer_idx, pacesetter_dict):
            continue
        if num_filters > target_deps[layer_idx]:
            result[layer_idx] = cluster_by_kmeans(kernel_value=named_kv.value, num_cluster=target_deps[layer_idx])
        elif num_filters < target_deps[layer_idx]:
            raise ValueError('wrong target dep')
    return result



def train_one_step(net, data, label, optimizer, criterion, param_name_to_merge_matrix, param_name_to_decay_matrix):
    pred = net(data)
    loss = criterion(pred, label)
    loss.backward()

    #TODO note: C-SGD works here
    for name, param in net.named_parameters():
        name = name.replace('module.', '')
        if name in param_name_to_merge_matrix:
            p_dim = param.dim()
            p_size = param.size()
            if p_dim == 4:
                param_mat = param.reshape(p_size[0], -1)
                g_mat = param.grad.reshape(p_size[0], -1)
            elif p_dim == 1:
                param_mat = param.reshape(p_size[0], 1)
                g_mat = param.grad.reshape(p_size[0], 1)
            else:
                assert p_dim == 2
                param_mat = param
                g_mat = param.grad
            csgd_gradient = param_name_to_merge_matrix[name].matmul(g_mat) + param_name_to_decay_matrix[name].matmul(param_mat)
            param.grad.copy_(csgd_gradient.reshape(p_size))

    optimizer.step()
    optimizer.zero_grad()

    acc, acc5 = torch_accuracy(pred, label, (1,5))
    return acc, acc5, loss


def sgd_optimizer(engine, cfg, model, no_l2_keywords, use_nesterov, keyword_to_lr_mult):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.base_lr
        weight_decay = cfg.weight_decay
        if "bias" in key or "bn" in key or "BN" in key:
            # lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.weight_decay_bias
            engine.echo('set weight_decay_bias={} for {}'.format(weight_decay, key))
        for kw in no_l2_keywords:
            if kw in key:
                weight_decay = 0
                engine.echo('NOTICE! weight decay = 0 for {} because {} in {}'.format(key, kw, key))
                break
        if 'bias' in key:
            apply_lr = 2 * lr
        else:
            apply_lr = lr
        if keyword_to_lr_mult is not None:
            for keyword, mult in keyword_to_lr_mult.items():
                if keyword in key:
                    apply_lr *= mult
                    engine.echo('multiply lr of {} by {}'.format(key, mult))
                    break
        params += [{"params": [value], "lr": apply_lr, "weight_decay": weight_decay}]
    # optimizer = torch.optim.Adam(params, lr)
    optimizer = torch.optim.SGD(params, lr, momentum=cfg.momentum, nesterov=use_nesterov)
    return optimizer

def get_optimizer(engine, cfg, model, no_l2_keywords, use_nesterov=False, keyword_to_lr_mult=None):
    return sgd_optimizer(engine, cfg, model, no_l2_keywords, use_nesterov=use_nesterov, keyword_to_lr_mult=keyword_to_lr_mult)

def get_criterion(cfg):
    return CrossEntropyLoss()


def csgd_train_main(
        local_rank,
        cfg:BaseConfigByEpoch, target_deps, succeeding_strategy, pacesetter_dict, centri_strength, pruned_weights,
        net=None, train_dataloader=None, val_dataloader=None, show_variables=False, convbuilder=None,
               init_hdf5=None, no_l2_keywords='depth', use_nesterov=False,
               load_weights_keyword=None,
               keyword_to_lr_mult=None,
               auto_continue=False,
save_hdf5_epochs=10000):


    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.tb_dir)
    clusters_save_path = os.path.join(cfg.output_dir, 'clusters.npy')

    with Engine(local_rank=local_rank) as engine:
        engine.setup_log(
            name='train', log_dir=cfg.output_dir, file_name='log.txt')

        # ----------------------------- build model ------------------------------
        if convbuilder is None:
            convbuilder = ConvBuilder(base_config=cfg)
        if net is None:
            net_fn = get_model_fn(cfg.dataset_name, cfg.network_type)
            model = net_fn(cfg, convbuilder)
        else:
            model = net
        model = model.cuda()
        # ----------------------------- model done ------------------------------

        # ---------------------------- prepare data -------------------------
        if train_dataloader is None:
            train_data = create_dataset(cfg.dataset_name, cfg.dataset_subset,
                                        cfg.global_batch_size, distributed=engine.distributed)
        if cfg.val_epoch_period > 0 and val_dataloader is None:
            val_data = create_dataset(cfg.dataset_name, 'val',
                                      global_batch_size=100, distributed=False)
        engine.echo('NOTE: Data prepared')
        engine.echo('NOTE: We have global_batch_size={} on {} GPUs, the allocated GPU memory is {}'.format(cfg.global_batch_size, torch.cuda.device_count(), torch.cuda.memory_allocated()))
        # ----------------------------- data done --------------------------------




        # ------------------------ parepare optimizer, scheduler, criterion -------
        if no_l2_keywords is None:
            no_l2_keywords = []
        if type(no_l2_keywords) is not list:
            no_l2_keywords = [no_l2_keywords]
        # For a target parameter, cancel its weight decay in optimizer, because the weight decay will be later encoded in the decay mat
        conv_idx = 0
        for k, v in model.named_parameters():
            if v.dim() != 4:
                continue
            print('prune {} from {} to {}'.format(conv_idx, target_deps[conv_idx], cfg.deps[conv_idx]))
            if target_deps[conv_idx] < cfg.deps[conv_idx]:
                no_l2_keywords.append(k.replace(KERNEL_KEYWORD, 'conv'))
                no_l2_keywords.append(k.replace(KERNEL_KEYWORD, 'bn'))
            conv_idx += 1
        print('no l2: ', no_l2_keywords)
        optimizer = get_optimizer(engine, cfg, model, no_l2_keywords=no_l2_keywords, use_nesterov=use_nesterov, keyword_to_lr_mult=keyword_to_lr_mult)
        scheduler = get_lr_scheduler(cfg, optimizer)
        criterion = get_criterion(cfg).cuda()
        # --------------------------------- done -------------------------------

        engine.register_state(
            scheduler=scheduler, model=model, optimizer=optimizer)

        if engine.distributed:
            torch.cuda.set_device(local_rank)
            engine.echo('Distributed training, device {}'.format(local_rank))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank],
                broadcast_buffers=False, )
        else:
            assert torch.cuda.device_count() == 1
            engine.echo('Single GPU training')

        if cfg.init_weights:
            engine.load_checkpoint(cfg.init_weights)
        if init_hdf5:
            engine.load_hdf5(init_hdf5, load_weights_keyword=load_weights_keyword)
        if auto_continue:
            assert cfg.init_weights is None
            engine.load_checkpoint(get_last_checkpoint(cfg.output_dir))
        if show_variables:
            engine.show_variables()

        #   ===================================== prepare the clusters and matrices for C-SGD ==========
        kernel_namedvalue_list = engine.get_all_conv_kernel_namedvalue_as_list()

        if os.path.exists(clusters_save_path):
            layer_idx_to_clusters = np.load(clusters_save_path, allow_pickle=True).item()
        else:
            if local_rank == 0:
                layer_idx_to_clusters = get_layer_idx_to_clusters(kernel_namedvalue_list=kernel_namedvalue_list,
                                                                  target_deps=target_deps,
                                                                  pacesetter_dict=pacesetter_dict)
                if pacesetter_dict is not None:
                    for follower_idx, pacesetter_idx in pacesetter_dict.items():
                        if pacesetter_idx in layer_idx_to_clusters:
                            layer_idx_to_clusters[follower_idx] = layer_idx_to_clusters[pacesetter_idx]

                np.save(clusters_save_path, layer_idx_to_clusters)
            else:
                while not os.path.exists(clusters_save_path):
                    time.sleep(10)
                    print('sleep, waiting for process 0 to calculate clusters')
                layer_idx_to_clusters = np.load(clusters_save_path, allow_pickle=True).item()


        param_name_to_merge_matrix = generate_merge_matrix_for_kernel(deps=cfg.deps,
                                                                      layer_idx_to_clusters=layer_idx_to_clusters,
                                                                      kernel_namedvalue_list=kernel_namedvalue_list)
        add_vecs_to_merge_mat_dicts(param_name_to_merge_matrix)
        param_name_to_decay_matrix = generate_decay_matrix_for_kernel_and_vecs(deps=cfg.deps,
                                                                               layer_idx_to_clusters=layer_idx_to_clusters,
                                                                               kernel_namedvalue_list=kernel_namedvalue_list,
                                                                               weight_decay=cfg.weight_decay,
                                                                               weight_decay_bias=cfg.weight_decay_bias,
                                                                               centri_strength=centri_strength)
        print(param_name_to_decay_matrix.keys())
        print(param_name_to_merge_matrix.keys())

        conv_idx = 0
        param_to_clusters = {}
        for k, v in model.named_parameters():
            if v.dim() != 4:
                continue
            if conv_idx in layer_idx_to_clusters:
                for clsts in layer_idx_to_clusters[conv_idx]:
                    if len(clsts) > 1:
                        param_to_clusters[v] = layer_idx_to_clusters[conv_idx]
                        break
            conv_idx += 1
        #   ============================================================================================





        # ------------ do training ---------------------------- #
        engine.log("\n\nStart training with pytorch version {}".format(torch.__version__))

        iteration = engine.state.iteration
        iters_per_epoch = num_iters_per_epoch(cfg)
        max_iters = iters_per_epoch * cfg.max_epochs
        tb_writer = SummaryWriter(cfg.tb_dir)
        tb_tags = ['Top1-Acc', 'Top5-Acc', 'Loss']

        model.train()

        done_epochs = iteration // iters_per_epoch
        last_epoch_done_iters = iteration % iters_per_epoch

        if done_epochs == 0 and last_epoch_done_iters == 0:
            engine.save_hdf5(os.path.join(cfg.output_dir, 'init.hdf5'))

        recorded_train_time = 0
        recorded_train_examples = 0

        collected_train_loss_sum = 0
        collected_train_loss_count = 0


        for epoch in range(done_epochs, cfg.max_epochs):

            if engine.distributed and hasattr(train_data, 'train_sampler'):
                train_data.train_sampler.set_epoch(epoch)

            if epoch == done_epochs:
                pbar = tqdm(range(iters_per_epoch - last_epoch_done_iters))
            else:
                pbar = tqdm(range(iters_per_epoch))

            if epoch == 0 and local_rank == 0:
                val_during_train(epoch=epoch, iteration=iteration, tb_tags=tb_tags, engine=engine, model=model,
                                 val_data=val_data, criterion=criterion, descrip_str='Init',
                                 dataset_name=cfg.dataset_name, test_batch_size=TEST_BATCH_SIZE,
                                 tb_writer=tb_writer)


            top1 = AvgMeter()
            top5 = AvgMeter()
            losses = AvgMeter()
            discrip_str = 'Epoch-{}/{}'.format(epoch, cfg.max_epochs)
            pbar.set_description('Train' + discrip_str)

            for _ in pbar:

                start_time = time.time()
                data, label = load_cuda_data(train_data, dataset_name=cfg.dataset_name)

                    # load_cuda_data(train_dataloader, cfg.dataset_name)
                data_time = time.time() - start_time

                train_net_time_start = time.time()
                acc, acc5, loss = train_one_step(model, data, label, optimizer, criterion,
                                                 param_name_to_merge_matrix=param_name_to_merge_matrix,
                                                 param_name_to_decay_matrix=param_name_to_decay_matrix)
                train_net_time_end = time.time()

                if iteration > TRAIN_SPEED_START * max_iters and iteration < TRAIN_SPEED_END * max_iters:
                    recorded_train_examples += cfg.global_batch_size
                    recorded_train_time += train_net_time_end - train_net_time_start

                scheduler.step()

                for module in model.modules():
                    if hasattr(module, 'set_cur_iter'):
                        module.set_cur_iter(iteration)

                if iteration % cfg.tb_iter_period == 0 and engine.world_rank == 0:
                    for tag, value in zip(tb_tags, [acc.item(), acc5.item(), loss.item()]):
                        tb_writer.add_scalars(tag, {'Train': value}, iteration)
                    deviation_sum = 0
                    for param, clusters in param_to_clusters.items():
                        pvalue = param.detach().cpu().numpy()
                        for cl in clusters:
                            if len(cl) == 1:
                                continue
                            selected = pvalue[cl, :, :, :]
                            mean_kernel = np.mean(selected, axis=0, keepdims=True)
                            diff = selected - mean_kernel
                            deviation_sum += np.sum(diff ** 2)
                    tb_writer.add_scalars('deviation_sum', {'Train': deviation_sum}, iteration)


                top1.update(acc.item())
                top5.update(acc5.item())
                losses.update(loss.item())

                if epoch >= cfg.max_epochs - COLLECT_TRAIN_LOSS_EPOCHS:
                    collected_train_loss_sum += loss.item()
                    collected_train_loss_count += 1

                pbar_dic = OrderedDict()
                pbar_dic['data-time'] = '{:.2f}'.format(data_time)
                pbar_dic['cur_iter'] = iteration
                pbar_dic['lr'] = scheduler.get_lr()[0]
                pbar_dic['top1'] = '{:.5f}'.format(top1.mean)
                pbar_dic['top5'] = '{:.5f}'.format(top5.mean)
                pbar_dic['loss'] = '{:.5f}'.format(losses.mean)
                pbar.set_postfix(pbar_dic)

                iteration += 1

                if iteration >= max_iters or iteration % cfg.ckpt_iter_period == 0:
                    engine.update_iteration(iteration)
                    if (not engine.distributed) or (engine.distributed and engine.world_rank == 0):
                        engine.save_and_link_checkpoint(cfg.output_dir)

                if iteration >= max_iters:
                    break

            #   do something after an epoch?
            engine.update_iteration(iteration)
            engine.save_latest_ckpt(cfg.output_dir)

            if (epoch + 1) % save_hdf5_epochs == 0:
                engine.save_hdf5(os.path.join(cfg.output_dir, 'epoch-{}.hdf5'.format(epoch)))

            if local_rank == 0 and \
                    cfg.val_epoch_period > 0 and (epoch >= cfg.max_epochs - 10 or epoch % cfg.val_epoch_period == 0):
                val_during_train(epoch=epoch, iteration=iteration, tb_tags=tb_tags, engine=engine, model=model,
                                 val_data=val_data, criterion=criterion, descrip_str=discrip_str,
                                 dataset_name=cfg.dataset_name, test_batch_size=TEST_BATCH_SIZE, tb_writer=tb_writer)

            if iteration >= max_iters:
                break

        #   do something after the training
        if recorded_train_time > 0:
            exp_per_sec = recorded_train_examples / recorded_train_time
        else:
            exp_per_sec = 0
        engine.log(
            'TRAIN speed: from {} to {} iterations, batch_size={}, examples={}, total_net_time={:.4f}, examples/sec={}'
            .format(int(TRAIN_SPEED_START * max_iters), int(TRAIN_SPEED_END * max_iters), cfg.global_batch_size,
                    recorded_train_examples, recorded_train_time, exp_per_sec))
        if cfg.save_weights:
            engine.save_checkpoint(cfg.save_weights)
            print('NOTE: training finished, saved to {}'.format(cfg.save_weights))
        engine.save_hdf5(os.path.join(cfg.output_dir, 'finish.hdf5'))
        if collected_train_loss_count > 0:
            engine.log('TRAIN LOSS collected over last {} epochs: {:.6f}'.format(COLLECT_TRAIN_LOSS_EPOCHS,
                                                                             collected_train_loss_sum / collected_train_loss_count))

    if local_rank == 0:
        csgd_prune_and_save(engine=engine, layer_idx_to_clusters=layer_idx_to_clusters,
                        save_file=pruned_weights, succeeding_strategy=succeeding_strategy, new_deps=target_deps)