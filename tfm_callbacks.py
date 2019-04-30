from abc import abstractmethod
import tensorflow as tf
import numpy as np
from tf_utils import tf_get_gradient_by_var

class GradientHandler(object):

    @abstractmethod
    def handle_gradient(self, origin_grads_and_vars):
        pass


class MergeGradientHandler(GradientHandler):

    def __init__(self, model, layer_to_eqcls, l2_factor, diff_factor, transmat_gpu_idx=0,
                 exclude_l2_decay_keywords=None, bn_layer_to_eqcls=None, version=2, slow_on_vec=False):
        self.layer_to_eqcls = layer_to_eqcls
        self.model = model
        self.l2_factor = l2_factor
        self.diff_factor = diff_factor
        self.transmat_gpu_idx = transmat_gpu_idx
        self.exclude_l2_decay_keywords=exclude_l2_decay_keywords
        self.bn_layer_to_eqcls = bn_layer_to_eqcls
        self.version = version
        assert version == 2
        print('merge version: ', version)
        self.slow_on_vec =slow_on_vec


    def merge_gradient_v2(self, target_t_grad, eqcls):
        num_filters = target_t_grad.get_shape()[0]

        #   TODO weights are now deprecated
        weights = np.ones(num_filters)

        merge_trans_mat = np.zeros((num_filters, num_filters), dtype=np.float32)
        for eqc in eqcls:
            if len(eqc) == 1:
                merge_trans_mat[eqc[0], eqc[0]] = 1
                continue
            weights_sum = np.sum(np.array([weights[ee] for ee in eqc]))
            se = sorted(eqc)
            for ei in se:
                for ej in se:
                    merge_trans_mat[ei, ej] = weights[ej] / weights_sum
        with tf.device("/gpu:{}".format(self.transmat_gpu_idx)):
            gpu_trans_mat = tf.constant(merge_trans_mat, dtype=tf.float32, name='merge_trans_var')
            merged_gradient = tf.matmul(gpu_trans_mat, target_t_grad)
        return merged_gradient


    def add_decay_to_merged_gradient_v2(self, merged_gradient, target_t_var, eqcls, l2_factor, diff_factor):
        print('diff_factor =', diff_factor)
        num_filters = target_t_var.get_shape()[0]
        decay_trans_mat = np.zeros((num_filters, num_filters), dtype=np.float32)
        for eqc in eqcls:
            for ee in eqc:
                decay_trans_mat[ee, ee] = l2_factor + diff_factor
                for p in eqc:
                    decay_trans_mat[ee, p] += -diff_factor / len(eqc)
        with tf.device("/gpu:{}".format(self.transmat_gpu_idx)):
            gpu_trans_mat = tf.constant(decay_trans_mat, dtype=tf.float32, name='decay_trans_var')
            decayed_gradient = merged_gradient + tf.matmul(gpu_trans_mat, target_t_var)
        return decayed_gradient


    def handle_gradient(self, origin_grads_and_vars):
        assert self.version == 2
        return self.merge_and_decay_grads_v2(self.model, origin_grads_and_vars, self.layer_to_eqcls)


    def merge_and_decay_grads_v2(self, model, origin_grads_and_vars, layer_to_eqcls):
        kernels = model.get_kernel_tensors()
        origin_g_to_decayed_g = {}

        #   merge and decay the 1D parameters (i.e., bias, gamma, beta)
        #   we apply no L2 regularization on vecs (Caffe style)
        def md_1d(layer_idx, var, l2):
            if var is not None:
                origin_g = tf_get_gradient_by_var(origin_grads_and_vars, var)
                reshaped_g = tf.expand_dims(origin_g, 1)
                merged_g = self.merge_gradient_v2(reshaped_g, layer_to_eqcls[layer_idx])
                if self.slow_on_vec:
                    vec_diff_factor = self.diff_factor / 10
                    print('use slowgammabeta by 10 times')
                else:
                    vec_diff_factor = self.diff_factor

                decayed_g = self.add_decay_to_merged_gradient_v2(merged_g, tf.expand_dims(var, 1), layer_to_eqcls[layer_idx], l2, vec_diff_factor)
                origin_g_to_decayed_g[origin_g] = tf.squeeze(decayed_g)


        for layer_idx in layer_to_eqcls.keys():
            #   handle kernel
            origin_kv = kernels[layer_idx]
            origin_kg = tf_get_gradient_by_var(origin_grads_and_vars, origin_kv)
            if 'depthwise' in origin_kv.name:
                kv = tf.transpose(origin_kv, [0,1,3,2])
                kg = tf.transpose(origin_kg, [0,1,3,2])
            else:
                kv = origin_kv
                kg = origin_kg

            kv_shape = kv.get_shape()

            if len(kv_shape) == 4:
                reshaped_kg = tf.transpose(tf.reshape(kg, (-1, kv_shape[3])), [1,0])
                reshaped_k = tf.transpose(tf.reshape(kv, (-1, kv_shape[3])), [1,0])
            else:
                assert len(kv_shape) == 2
                print('handling gradients for fc kernel: ', kv.name)
                reshaped_kg = tf.transpose(kg, [1,0])
                reshaped_k = tf.transpose(kv, [1,0])

            merged_kg = self.merge_gradient_v2(reshaped_kg, layer_to_eqcls[layer_idx])
            apply_l2 = 0 if (self.exclude_l2_decay_keywords is not None and self.exclude_l2_decay_keywords in kv.name) else self.l2_factor
            decayed_kg = self.add_decay_to_merged_gradient_v2(merged_kg, reshaped_k, layer_to_eqcls[layer_idx], apply_l2, self.diff_factor)

            restored_kg = tf.transpose(decayed_kg, [1,0])
            restored_kg = tf.reshape(restored_kg, kv_shape)
            if 'depthwise' in origin_kv.name:
                restored_kg = tf.transpose(restored_kg, [0,1,3,2])

            origin_g_to_decayed_g[origin_kg] = restored_kg
            #   handlle other vectors
            bias_variable = model.get_bias_variable_for_kernel(layer_idx)
            md_1d(layer_idx, bias_variable, 0)
            gamma_variable = model.get_gamma_variable_for_kernel(layer_idx)
            md_1d(layer_idx, gamma_variable, 0)
            beta_variable = model.get_beta_variable_for_kernel(layer_idx)
            md_1d(layer_idx, beta_variable, 0)


        #   deal with the separate BN layers
        if self.bn_layer_to_eqcls is not None:

            def md_1d_bn(eqcls, var):
                if var is not None:
                    origin_g = tf_get_gradient_by_var(origin_grads_and_vars, var)
                    reshaped_g = tf.expand_dims(origin_g, 1)
                    merged_g = self.merge_gradient_v2(reshaped_g, eqcls)
                    if self.slow_on_vec:
                        vec_diff_factor = self.diff_factor / 10
                        print('use slowgammabeta by 10 times')
                    else:
                        vec_diff_factor = self.diff_factor
                    decayed_g = self.add_decay_to_merged_gradient_v2(merged_g, tf.expand_dims(var, 1), eqcls, 0, vec_diff_factor)
                    origin_g_to_decayed_g[origin_g] = tf.squeeze(decayed_g)

            beta_tensors = model.get_beta_tensors()
            gamma_tensors = model.get_gamma_tensors()
            for bn_layer_idx, bn_eqcls in self.bn_layer_to_eqcls.items():
                md_1d_bn(bn_eqcls, beta_tensors[bn_layer_idx])
                md_1d_bn(bn_eqcls, gamma_tensors[bn_layer_idx])

        result = []
        merged_cnt = 0
        for (g, v) in origin_grads_and_vars:
            if g in origin_g_to_decayed_g:
                result.append((origin_g_to_decayed_g[g], v))
                merged_cnt += 1
            else:
                result.append((g, v))
        print(merged_cnt, 'gradients merged')
        return result



class Callback(object):

    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_step(self, step):
        pass

    def after_step(self, step):
        pass

class CallbackList(object):

    def __init__(self, callback_list):
        if type(callback_list) is list:
            self.callback_list = callback_list
        else:
            self.callback_list = [callback_list]

    def before_train(self):
        for c in self.callback_list:
            c.before_train()

    def after_train(self):
        for c in self.callback_list:
            c.after_train()

    def before_step(self, step):
        for c in self.callback_list:
            c.before_step(step)

    def after_step(self, step):
        for c in self.callback_list:
            c.after_step(step)