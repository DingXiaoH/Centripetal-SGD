from tf_utils import *
from sklearn.cluster import KMeans

def _sk_cluster(model, layer_idx, num_eqcls, cluster_class):
    x = model.get_value(model.get_kernel_tensors()[layer_idx])
    if x.ndim == 4:
        x = np.reshape(x, (-1, x.shape[3]))
    x = np.transpose(x, [1,0])

    if num_eqcls == x.shape[0]:
        result = [[i] for i in range(num_eqcls)]
        return result

    km = cluster_class(n_clusters=num_eqcls)
    km.fit(x)
    result = []
    for j in range(num_eqcls):
        result.append([])
    for i, c in enumerate(km.labels_):
        result[c].append(i)
    #   do check
    for r in result:
        assert len(r) > 0
    return result

def calculate_eqcls_by_kmeans(model, layer_idx, num_eqcls):
    print('applying kmeans clustering')
    return _sk_cluster(model, layer_idx, num_eqcls, KMeans)


def calculate_eqcls_evenly(filters, num_eqcls):
    result = []
    min_filters_per_eqcl = filters // num_eqcls
    left = filters % num_eqcls
    cur_filter_idx = 0
    for i in range(num_eqcls):
        if left > 0:
            left -= 1
            nb_filters_cur_eqcl = min_filters_per_eqcl + 1
        else:
            nb_filters_cur_eqcl = min_filters_per_eqcl
        cur_eqcl = [cur_filter_idx + p for p in range(nb_filters_cur_eqcl)]
        cur_filter_idx += nb_filters_cur_eqcl
        result.append(cur_eqcl)
    return result

def calculate_eqcls_biasly(filters, num_eqcls):
    result = []

    num_filters_in_first_eqcl = filters - num_eqcls + 1

    first_eqcl = [i for i in range(num_filters_in_first_eqcl)]
    result.append(first_eqcl)
    for i in range(num_eqcls - 1):
        result.append([i + num_filters_in_first_eqcl])

    return result


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



def eqcls_indexes_to_delete(eqcls):
    result = []
    for eqc in eqcls:
        eqcl = list(sorted(eqc))
        result += eqcl[1:]
    return result

def num_filters_in_eqcls(eqcls):
    num = 0
    max_idx = 0
    for eqc in eqcls:
        num += len(eqc)
        max_idx = max(max_idx, max(eqc))
    assert max_idx == num - 1
    return num

def shift_eqcls(eqcls, offset):
    result = []
    for eqc in eqcls:
        new_eqc = [offset + e for e in eqc]
        result.append(new_eqc)
    return result

def calculate_bn_eqcls_dc40(conv_layer_to_eqcls):
    bn_layer_idx_to_eqcls = {0 : list(conv_layer_to_eqcls[0])}
    def calc_bn_eqcls(layer_range):
        for i in layer_range:
            last_layer_eqcls = bn_layer_idx_to_eqcls[i - 1]
            num_filters_in_last_layer_eqcls = num_filters_in_eqcls(last_layer_eqcls)
            cur_layer_eqcls = last_layer_eqcls + shift_eqcls(list(conv_layer_to_eqcls[i]),
                offset=num_filters_in_last_layer_eqcls)
            bn_layer_idx_to_eqcls[i] = cur_layer_eqcls
    calc_bn_eqcls(range(1, 13))
    bn_layer_idx_to_eqcls[13] = list(conv_layer_to_eqcls[13])
    calc_bn_eqcls(range(14, 26))
    bn_layer_idx_to_eqcls[26] = list(conv_layer_to_eqcls[26])
    calc_bn_eqcls(range(27, 39))
    return bn_layer_idx_to_eqcls





def tfm_prune_filters_and_save_dc40(model, conv_layer_to_eqcls, bn_layer_to_eqcls, save_file, new_deps=None):

    kernel_tensors = model.get_kernel_tensors()
    mu_tensors = model.get_moving_mean_tensors()
    var_tensors = model.get_moving_variance_tensors()
    beta_tensors = model.get_beta_tensors()
    gamma_tensors = model.get_gamma_tensors()
    assert len(gamma_tensors) == 39
    assert len(conv_layer_to_eqcls) == 39

    result = {}

    #   prune all the conv layers, NO NEED to adjust following layers
    for layer_idx, eqcls in conv_layer_to_eqcls.items():
        kernel = kernel_tensors[layer_idx]
        kv = model.get_value(kernel)
        conv_idxes_to_delete = eqcls_indexes_to_delete(eqcls)
        pruned_kv = delete_or_keep(kv, idxes=conv_idxes_to_delete, axis=3)
        result[kernel.name] = pruned_kv

    #   prune bn layers and re-construct the following conv layers (if exists)
    for i in range(0, 39):
        bn_eqcls = bn_layer_to_eqcls[i]
        mu = mu_tensors[i]
        var = var_tensors[i]
        beta = beta_tensors[i]
        gamma = gamma_tensors[i]
        bn_eqcls_to_delete = eqcls_indexes_to_delete(bn_eqcls)

        result[mu.name] = delete_or_keep(model.get_value(mu), idxes=bn_eqcls_to_delete)
        result[var.name] = delete_or_keep(model.get_value(var), idxes=bn_eqcls_to_delete)
        result[beta.name] = delete_or_keep(model.get_value(beta), idxes=bn_eqcls_to_delete)
        result[gamma.name] = delete_or_keep(model.get_value(gamma), idxes=bn_eqcls_to_delete)

        if i < 38:
            follow_kernel = kernel_tensors[i + 1]
            follow_kernel_value = result[follow_kernel.name]
            for eqcl in bn_eqcls:
                if len(eqcl) == 1:
                    continue
                eqc = np.array(sorted(eqcl))
                selected_k_follow = follow_kernel_value[:, :, eqc, :]
                aggregated_k_follow = np.sum(selected_k_follow, axis=2)
                follow_kernel_value[:, :, eqc[0], :] = aggregated_k_follow
            result[follow_kernel.name] = delete_or_keep(follow_kernel_value, idxes=bn_eqcls_to_delete, axis=2)


    #   deal with the fc layer
    fc_kernel = kernel_tensors[-1]
    fc_value = model.get_value(fc_kernel)
    fc_indexes_to_delete = []
    origin_last_bn_width = num_filters_in_eqcls(bn_layer_to_eqcls[38])
    corresponding_neurons_per_kernel = fc_value.shape[0] // origin_last_bn_width
    base = np.arange(0, corresponding_neurons_per_kernel * origin_last_bn_width, origin_last_bn_width)
    for eqcl in bn_layer_to_eqcls[38]:
        if len(eqcl) == 1:
            continue
        se = sorted(eqcl)
        for i in se[1:]:
            fc_indexes_to_delete.append(base + i)
        to_concat = []
        for i in se:
            corresponding_neurons_idxes = base + i
            to_concat.append(np.expand_dims(fc_value[corresponding_neurons_idxes, :], axis=0))
        merged = np.sum(np.concatenate(to_concat, axis=0), axis=0)
        reserved_idxes = base + se[0]
        fc_value[reserved_idxes, :] = merged
    if len(fc_indexes_to_delete) > 0:
        fc_value = delete_or_keep(fc_value, np.concatenate(fc_indexes_to_delete, axis=0), axis=0)
    result[fc_kernel.name] = fc_value
    key_variables = model.get_key_variables()
    for var in key_variables:
        if var.name not in result:
            result[var.name] = model.get_value(var)
    if new_deps is not None:
        result['deps'] = new_deps
    print('save {} varialbes to {} after pruning filters'.format(len(result), save_file))
    if save_file.endswith('npy'):
        np.save(save_file, result)
    else:
        save_hdf5(result, save_file)


#   assume that the filters have been merged and the following variables have been adjusted
def tfm_prune_filters_and_save(model, layer_to_eqcls, save_file, fc_layer_idxes,
                               subsequent_strategy, layer_idx_to_follow_offset={},
                               fc_neurons_per_kernel=None, new_deps=None):
    result = dict()
    number_filters_seen = 0
    num_filters_alike = 0

    if subsequent_strategy is None:
        subsequent_map = None
    elif subsequent_strategy == 'simple':
        subsequent_map = {idx : (idx+1) for idx in layer_to_eqcls.keys()}
    else:
        subsequent_map = subsequent_strategy
    if type(fc_layer_idxes) is not list:
        fc_layer_idxes = [fc_layer_idxes]

    kernels = model.get_kernel_tensors()

    for layer_idx, eqcls in layer_to_eqcls.items():

        kernel_tensor = kernels[layer_idx]
        print('cur kernel name:', kernel_tensor.name)
        bias_tensor = model.get_bias_variable_for_kernel(layer_idx)
        beta_tensor = model.get_beta_variable_for_kernel(layer_idx)
        gamma_tensor = model.get_gamma_variable_for_kernel(layer_idx)
        moving_mean_tensor = model.get_moving_mean_variable_for_kernel(layer_idx)
        moving_variance_tensor = model.get_moving_variance_variable_for_kernel(layer_idx)

        if kernel_tensor.name in result:
            kernel_value = result[kernel_tensor.name]
        else:
            kernel_value = model.get_value(kernel_tensor)

        if subsequent_map is None or layer_idx not in subsequent_map:
            indexes_to_delete = []
            for eqcl in eqcls:
                number_filters_seen += len(eqcl)
                if len(eqcl) == 1:
                    continue
                num_filters_alike += len(eqcl)
                indexes_to_delete += eqcl[1:]
        else:
            follows = subsequent_map[layer_idx]
            print('{} follows {}'.format(follows, layer_idx))
            if type(follows) is not list:
                follows = [follows]
            for follow_idx in follows:
                follow_kernel_tensor = kernels[follow_idx]
                if follow_kernel_tensor.name in result:
                    kvf = result[follow_kernel_tensor.name]
                else:
                    kvf = model.get_value(follow_kernel_tensor)
                print('following kernel name: ', follow_kernel_tensor.name, 'origin shape: ', kvf.shape)

                if follow_idx in fc_layer_idxes:
                    offset = layer_idx_to_follow_offset.get(layer_idx, 0)
                    if offset > 0:
                        print('offset,',offset)
                    conv_indexes_to_delete = []
                    fc_indexes_to_delete = []
                    # assert kvf.shape[0] % kernel_value.shape[3] == 0

                    if fc_neurons_per_kernel is None:
                        conv_deps = kernel_value.shape[3] + offset
                        corresponding_neurons_per_kernel = kvf.shape[0] // conv_deps
                    else:
                        corresponding_neurons_per_kernel=fc_neurons_per_kernel
                        conv_deps = kvf.shape[0] // corresponding_neurons_per_kernel
                    print('total conv deps:', conv_deps, corresponding_neurons_per_kernel, 'neurons per kernel')

                    base = np.arange(offset, corresponding_neurons_per_kernel*conv_deps+offset, conv_deps)
                    for eqcl in eqcls:
                        number_filters_seen += len(eqcl)
                        if len(eqcl) == 1:
                            continue
                        num_filters_alike += len(eqcl)
                        conv_indexes_to_delete += eqcl[1:]
                        for i in eqcl[1:]:
                            fc_indexes_to_delete.append(base + i)
                        to_concat = []
                        for i in eqcl:
                            corresponding_neurons_idxes = base + i
                            to_concat.append(np.expand_dims(kvf[corresponding_neurons_idxes, :], axis=0))
                        merged = np.sum(np.concatenate(to_concat, axis=0), axis=0)
                        reserved_idxes = base + eqcl[0]
                        kvf[reserved_idxes, :] = merged
                    if len(fc_indexes_to_delete) > 0:
                        kvf = delete_or_keep(kvf, np.concatenate(fc_indexes_to_delete, axis=0), axis=0)
                        result[follow_kernel_tensor.name] = kvf
                        print('shape of pruned following kernel: ', kvf.shape)
                    indexes_to_delete = conv_indexes_to_delete
                else:
                    offset = layer_idx_to_follow_offset.get(layer_idx, 0)
                    indexes_to_delete = []
                    for eqcl in eqcls:
                        number_filters_seen += len(eqcl)
                        if len(eqcl) == 1:
                            continue
                        num_filters_alike += len(eqcl)
                        indexes_to_delete += eqcl[1:]
                        eqc = np.array(eqcl)
                        selected_k_follow = kvf[:, :, eqc+offset, :]
                        aggregated_k_follow = np.sum(selected_k_follow, axis=2)
                        kvf[:, :, eqcl[0]+offset, :] = aggregated_k_follow
                    if 'depth' in follow_kernel_tensor.name:
                        print('skip adding up and pruning the following layer, because it is a depthwise layer')
                    else:
                        follow_indexes_to_delete = [offset + p for p in indexes_to_delete]
                        kvf = delete_or_keep(kvf, follow_indexes_to_delete, axis=2)
                        result[follow_kernel_tensor.name] = kvf
                        print('shape of pruned following kernel: ', kvf.shape)
        if 'depth' in kernel_tensor.name:
            kernel_value_after_pruned = delete_or_keep(kernel_value, indexes_to_delete, axis=2)
        else:
            kernel_value_after_pruned = delete_or_keep(kernel_value, indexes_to_delete, axis=3)
        result[kernel_tensor.name] = kernel_value_after_pruned
        if bias_tensor is not None:
            bias_value = delete_or_keep(model.get_value(bias_tensor), indexes_to_delete)
            result[bias_tensor.name] = bias_value
        if moving_mean_tensor is not None:
            moving_mean_value = delete_or_keep(model.get_value(moving_mean_tensor), indexes_to_delete)
            result[moving_mean_tensor.name] = moving_mean_value
        if moving_variance_tensor is not None:
            moving_variance_value = delete_or_keep(model.get_value(moving_variance_tensor), indexes_to_delete)
            result[moving_variance_tensor.name] = moving_variance_value
        if beta_tensor is not None:
            beta_value = delete_or_keep(model.get_value(beta_tensor), indexes_to_delete)
            result[beta_tensor.name] = beta_value
        if gamma_tensor is not None:
            gamma_value = delete_or_keep(model.get_value(gamma_tensor), indexes_to_delete)
            result[gamma_tensor.name] = gamma_value
        print('kernel name: ', kernel_tensor.name)
        print(
            'removed filters. {} filters seen. {} filters alike. shape of origin kernel {}, shape of pruned kernel {}'
            .format(number_filters_seen, num_filters_alike, kernel_value.shape, kernel_value_after_pruned.shape))

    key_variables = model.get_key_variables()
    for var in key_variables:
        if var.name not in result:
            result[var.name] = model.get_value(var)
    if new_deps is not None:
        result['deps'] = new_deps
    print('save {} varialbes to {} after pruning filters'.format(len(result), save_file))
    if save_file.endswith('npy'):
        np.save(save_file, result)
    else:
        save_hdf5(result, save_file)



def delete_or_keep(array, idxes, axis=None):
    if len(idxes) > 0:
        return np.delete(array, idxes, axis=axis)
    else:
        return array


#   items can be a list of tensors or a numpy array
def _weighted_mean(items, weights):
    assert len(items) == len(weights)
    a_weights = np.array(weights)
    weights_sum = np.sum(a_weights)
    if weights_sum == 0:
        normalized = np.zeros_like(a_weights)
    else:
        normalized = a_weights / weights_sum
    sum = 0
    for item, weight in zip(items, normalized):
        sum += item * weight
    return sum


