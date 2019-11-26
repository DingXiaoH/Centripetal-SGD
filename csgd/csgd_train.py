from torch.utils.tensorboard import SummaryWriter
from base_config import BaseConfigByEpoch
from model_map import get_model_fn
from dataset import create_dataset, load_cuda_data, num_iters_per_epoch
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
import numpy as np
from ding_test import run_eval
from csgd.csgd_prune import csgd_prune_and_save
from sklearn.cluster import KMeans


TRAIN_SPEED_START = 0.1
TRAIN_SPEED_END = 0.2

KERNEL_KEYWORD = 'conv.weight'

def add_vecs_to_mat_dicts(param_name_to_merge_matrix):
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

def generate_decay_matrix_for_kernel_and_vecs(deps, layer_idx_to_clusters, kernel_namedvalue_list, weight_decay, centri_strength):
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
        result[kernel_namedvalue_list[layer_idx].name.replace(KERNEL_KEYWORD, 'bn.bias')] = kernel_mat
        result[kernel_namedvalue_list[layer_idx].name.replace(KERNEL_KEYWORD, 'conv.bias')] = kernel_mat

    #   for the vec params (bias, beta and gamma), we use 0.1 * centripetal strength
    for layer_idx, clusters in layer_idx_to_clusters.items():
        num_filters = deps[layer_idx]
        decay_trans_mat = np.zeros((num_filters, num_filters), dtype=np.float32)
        for clst in clusters:
            sc = sorted(clst)
            for ee in sc:
                # Note: using smaller centripetal strength on the scaling factor of BN improve the performance in some of the cases
                decay_trans_mat[ee, ee] = weight_decay + centri_strength * 0.1
                for p in sc:
                    decay_trans_mat[ee, p] += -centri_strength * 0.1 / len(clst)
        vec_mat = torch.from_numpy(decay_trans_mat).cuda()

        result[kernel_namedvalue_list[layer_idx].name.replace(KERNEL_KEYWORD, 'bn.weight')] = vec_mat

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

    for name, param in net.named_parameters():
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


def sgd_optimizer(cfg, model, use_nesterov):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.base_lr
        if "bias" in key or "bn" in key or "BN" in key:
            # lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.weight_decay_bias
            print('set weight_decay_bias={} for {}'.format(weight_decay, key))
        if 'bias' in key:
            apply_lr = 2 * lr
        else:
            apply_lr = lr
        params += [{"params": [value], "lr": apply_lr, "weight_decay": 0}]
    optimizer = torch.optim.SGD(params, lr, momentum=cfg.momentum, nesterov=use_nesterov)
    return optimizer


def get_optimizer(cfg, model, use_nesterov=False):
    return sgd_optimizer(cfg, model, use_nesterov=use_nesterov)

def get_criterion(cfg):
    return CrossEntropyLoss()

def csgd_train_and_prune(cfg:BaseConfigByEpoch,
                        target_deps, centri_strength, pacesetter_dict, succeeding_strategy, pruned_weights,
                         net=None, train_dataloader=None, val_dataloader=None, show_variables=False, convbuilder=None, beginning_msg=None,
               init_hdf5=None, no_l2_keywords=None, use_nesterov=False, tensorflow_style_init=False):

    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.tb_dir)
    clusters_save_path = os.path.join(cfg.output_dir, 'clusters.npy')

    with Engine() as engine:

        is_main_process = (engine.world_rank == 0) #TODO correct?

        logger = engine.setup_log(
            name='train', log_dir=cfg.output_dir, file_name='log.txt')

        # -- typical model components model, opt,  scheduler,  dataloder --#
        if net is None:
            net = get_model_fn(cfg.dataset_name, cfg.network_type)

        if convbuilder is None:
            convbuilder = ConvBuilder(base_config=cfg)

        model = net(cfg, convbuilder).cuda()

        if train_dataloader is None:
            train_dataloader = create_dataset(cfg.dataset_name, cfg.dataset_subset, cfg.global_batch_size)
        if cfg.val_epoch_period > 0 and val_dataloader is None:
            val_dataloader = create_dataset(cfg.dataset_name, 'val', batch_size=100)    #TODO 100?

        print('NOTE: Data prepared')
        print('NOTE: We have global_batch_size={} on {} GPUs, the allocated GPU memory is {}'.format(cfg.global_batch_size, torch.cuda.device_count(), torch.cuda.memory_allocated()))

        optimizer = get_optimizer(cfg, model, use_nesterov=use_nesterov)
        scheduler = get_lr_scheduler(cfg, optimizer)
        criterion = get_criterion(cfg).cuda()

        # model, optimizer = amp.initialize(model, optimizer, opt_level="O0")

        engine.register_state(
            scheduler=scheduler, model=model, optimizer=optimizer, cfg=cfg)

        if engine.distributed:
            print('Distributed training, engine.world_rank={}'.format(engine.world_rank))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[engine.world_rank],
                broadcast_buffers=False, )
            # model = DistributedDataParallel(model, delay_allreduce=True)
        elif torch.cuda.device_count() > 1:
            print('Single machine multiple GPU training')
            model = torch.nn.parallel.DataParallel(model)


        if tensorflow_style_init:
            for k, v in model.named_parameters():
                if v.dim() in [2, 4]:
                    torch.nn.init.xavier_uniform_(v)
                    print('init {} as xavier_uniform'.format(k))
                if 'bias' in k and 'bn' not in k.lower():
                    torch.nn.init.zeros_(v)
                    print('init {} as zero'.format(k))

        if cfg.init_weights:
            engine.load_checkpoint(cfg.init_weights)

        if init_hdf5:
            engine.load_hdf5(init_hdf5)

        kernel_namedvalue_list = engine.get_all_conv_kernel_namedvalue_as_list()

        if os.path.exists(clusters_save_path):
            layer_idx_to_clusters = np.load(clusters_save_path).item()
        else:
            layer_idx_to_clusters = get_layer_idx_to_clusters(kernel_namedvalue_list=kernel_namedvalue_list,
                                                              target_deps=target_deps, pacesetter_dict=pacesetter_dict)
            if pacesetter_dict is not None:
                for follower_idx, pacesetter_idx in pacesetter_dict.items():
                    if pacesetter_idx in layer_idx_to_clusters:
                        layer_idx_to_clusters[follower_idx] = layer_idx_to_clusters[pacesetter_idx]

            np.save(clusters_save_path, layer_idx_to_clusters)

        csgd_save_file = os.path.join(cfg.output_dir, 'finish.hdf5')


        if os.path.exists(csgd_save_file):
            engine.load_hdf5(csgd_save_file)
        else:
            param_name_to_merge_matrix = generate_merge_matrix_for_kernel(deps=cfg.deps,
                                                                          layer_idx_to_clusters=layer_idx_to_clusters,
                                                                          kernel_namedvalue_list=kernel_namedvalue_list)
            param_name_to_decay_matrix = generate_decay_matrix_for_kernel_and_vecs(deps=cfg.deps,
                                                                          layer_idx_to_clusters=layer_idx_to_clusters,
                                                                          kernel_namedvalue_list=kernel_namedvalue_list,
                                                                          weight_decay=cfg.weight_decay,
                                                                          centri_strength=centri_strength)
            # if pacesetter_dict is not None:
            #     for follower_idx, pacesetter_idx in pacesetter_dict.items():
            #         follower_kernel_name = kernel_namedvalue_list[follower_idx].name
            #         pacesetter_kernel_name = kernel_namedvalue_list[follower_idx].name
            #         if pacesetter_kernel_name in param_name_to_merge_matrix:
            #             param_name_to_merge_matrix[follower_kernel_name] = param_name_to_merge_matrix[
            #                 pacesetter_kernel_name]
            #             param_name_to_decay_matrix[follower_kernel_name] = param_name_to_decay_matrix[
            #                 pacesetter_kernel_name]

            add_vecs_to_mat_dicts(param_name_to_merge_matrix)

            if show_variables:
                engine.show_variables()

            if beginning_msg:
                engine.log(beginning_msg)

            logger.info("\n\nStart training with pytorch version {}".format(torch.__version__))

            iteration = engine.state.iteration
            # done_epochs = iteration // num_train_examples_per_epoch(cfg.dataset_name)
            iters_per_epoch = num_iters_per_epoch(cfg)
            max_iters = iters_per_epoch * cfg.max_epochs
            tb_writer = SummaryWriter(cfg.tb_dir)
            tb_tags = ['Top1-Acc', 'Top5-Acc', 'Loss']

            model.train()

            done_epochs = iteration // iters_per_epoch

            engine.save_hdf5(os.path.join(cfg.output_dir, 'init.hdf5'))

            recorded_train_time = 0
            recorded_train_examples = 0

            for epoch in range(done_epochs, cfg.max_epochs):

                pbar = tqdm(range(iters_per_epoch))
                top1 = AvgMeter()
                top5 = AvgMeter()
                losses = AvgMeter()
                discrip_str = 'Epoch-{}/{}'.format(epoch, cfg.max_epochs)
                pbar.set_description('Train' + discrip_str)

                if cfg.val_epoch_period > 0 and epoch % cfg.val_epoch_period == 0:
                    model.eval()
                    val_iters = 500 if cfg.dataset_name == 'imagenet' else 100  # use batch_size=100 for val on ImagenNet and CIFAR
                    eval_dict, _ = run_eval(val_dataloader, val_iters, model, criterion, discrip_str,
                                            dataset_name=cfg.dataset_name)
                    val_top1_value = eval_dict['top1'].item()
                    val_top5_value = eval_dict['top5'].item()
                    val_loss_value = eval_dict['loss'].item()
                    for tag, value in zip(tb_tags, [val_top1_value, val_top5_value, val_loss_value]):
                        tb_writer.add_scalars(tag, {'Val': value}, iteration)
                    engine.log(
                        'validate at epoch {}, top1={:.5f}, top5={:.5f}, loss={:.6f}'.format(epoch, val_top1_value,
                                                                                             val_top5_value,
                                                                                             val_loss_value))
                    model.train()

                for _ in pbar:

                    start_time = time.time()
                    data, label = load_cuda_data(train_dataloader, cfg.dataset_name)
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

                    if iteration % cfg.tb_iter_period == 0 and is_main_process:
                        for tag, value in zip(tb_tags, [acc.item(), acc5.item(), loss.item()]):
                            tb_writer.add_scalars(tag, {'Train': value}, iteration)

                    top1.update(acc.item())
                    top5.update(acc5.item())
                    losses.update(loss.item())

                    pbar_dic = OrderedDict()
                    pbar_dic['data-time'] = '{:.2f}'.format(data_time)
                    pbar_dic['cur_iter'] = iteration
                    pbar_dic['lr'] = scheduler.get_lr()[0]
                    pbar_dic['top1'] = '{:.5f}'.format(top1.mean)
                    pbar_dic['top5'] = '{:.5f}'.format(top5.mean)
                    pbar_dic['loss'] = '{:.5f}'.format(losses.mean)
                    pbar.set_postfix(pbar_dic)

                    if iteration >= max_iters or iteration % cfg.ckpt_iter_period == 0:
                        engine.update_iteration(iteration)
                        if (not engine.distributed) or (engine.distributed and is_main_process):
                            engine.save_and_link_checkpoint(cfg.output_dir)

                    iteration += 1
                    if iteration >= max_iters:
                        break

                #   do something after an epoch?
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


        csgd_prune_and_save(engine=engine, layer_idx_to_clusters=layer_idx_to_clusters,
                            save_file=pruned_weights, succeeding_strategy=succeeding_strategy, new_deps=target_deps)