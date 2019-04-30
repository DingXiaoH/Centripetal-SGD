from tf_utils import *
import copy

class TFModel(object):

    def __init__(self, dataset, inference_fn, mode, batch_size, image_size, need_validation=False, validation_batch_size=64, l2_factor=0.0, deps=None,
                 need_shuffle=False):
        self.dataset = dataset
        self.mode = mode
        preprocessor_type = dataset.preprocessor_type()
        if need_shuffle:
            self.data_preprocessor = preprocessor_type(mode, dataset, image_size=image_size,
                                                       batch_size=batch_size, num_preprocess_threads=8, need_shuffle=True)
        else:
            self.data_preprocessor = preprocessor_type(mode, dataset, image_size=image_size,
                batch_size=batch_size, num_preprocess_threads=8)
        self.graph = tf.Graph()
        self.batch_size = batch_size
        self.inference_fn = inference_fn
        self.need_initialization = True
        self.l2_factor=l2_factor
        self.queue_runner_started = False
        self.image_size = image_size
        if deps is not None:
            self.deps = np.array(deps)
        else:
            self.deps = None

        if need_validation:
            assert mode != 'eval'
            self.support_validation = True
            self.validation_has_initialized = False
            self.validation_batch_size = validation_batch_size
            val_dataset = copy.deepcopy(dataset)
            val_dataset.subset = 'validation'
            self.num_validation_examples = val_dataset.num_examples_per_epoch()
            self.val_data_preprocessor = preprocessor_type('eval', val_dataset,
                image_size=image_size, batch_size=validation_batch_size, num_preprocess_threads=4, num_readers=1)

        with self.graph.as_default():
            # self.var_training = tf.get_variable('var_training', shape=(), dtype=tf.bool, trainable=False)
            self.input_images, self.input_labels = self.data_preprocessor.get_batch_input_tensors()
            # just compile
            self.output = inference_fn(self.input_images)   # for eval only
            self.name_to_variables ={}
            for v in self.get_global_variables():
                self.name_to_variables[v.name] = v
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

    def initialize(self):
        # if not self.need_initialization:
        #     return
        with self.graph.as_default():
            init = tf.global_variables_initializer()
            self.sess.run(init)
        self.need_initialization = False


    def _get_variables_by_keyword(self, keyword):
        result = []
        for t in self.get_global_variables():
            if keyword in t.name:
                result.append(t)
        return result

    def get_variable_by_name(self, name):
        return self.name_to_variables[name]

    def get_global_variables(self):
        return self.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    def get_trainable_variables(self):
        return self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    def get_kernel_tensors(self):
        return self._get_variables_by_keyword('kernel')

    def get_bias_tensors(self):
        return self._get_variables_by_keyword('bias')

    def get_moving_mean_tensors(self):
        return self._get_variables_by_keyword('moving_mean')

    def get_moving_variance_tensors(self):
        return self._get_variables_by_keyword('moving_variance')

    def get_gamma_tensors(self):
        return self._get_variables_by_keyword('gamma')

    def get_beta_tensors(self):
        return self._get_variables_by_keyword('beta')


    def _get_variable_for_kernel(self, kernel_idx, keyword):
        variable_name = self.get_kernel_tensors()[kernel_idx].name.replace('kernel', keyword)
        if variable_name in self.name_to_variables:
            return self.name_to_variables[variable_name]
        else:
            possible_name = variable_name.replace('/'+keyword, '_bn/'+keyword)
            if possible_name in self.name_to_variables:
                return self.name_to_variables[possible_name]
            else:
                return None

    def get_bias_variable_for_kernel(self, kernel_idx):
        return self._get_variable_for_kernel(kernel_idx, 'bias')

    def get_moving_mean_variable_for_kernel(self, kernel_idx):
        return self._get_variable_for_kernel(kernel_idx, 'moving_mean')

    def get_moving_variance_variable_for_kernel(self, kernel_idx):
        return self._get_variable_for_kernel(kernel_idx, 'moving_variance')

    def get_beta_variable_for_kernel(self, kernel_idx):
        return self._get_variable_for_kernel(kernel_idx, 'beta')

    def get_gamma_variable_for_kernel(self, kernel_idx):
        return self._get_variable_for_kernel(kernel_idx, 'gamma')

    def calculate_l2_loss(self, tensors):
        l2_sum = 0.
        for t in tensors:
            l2_sum += 0.5 * tf.reduce_sum(tf.square(t))
        return l2_sum




    def get_pred_loss_and_acc(self, image_batch, label_batch):
        tf.get_variable_scope().reuse_variables()
        logits = self.inference_fn(image_batch)
        pred = tf.argmax(logits, 1)
        equ = tf.equal(tf.cast(label_batch, tf.int32), tf.cast(pred, tf.int32))
        acc_op = tf.reduce_mean(tf.cast(equ, tf.float32))

        sparse_labels = tf.reshape(label_batch, [self.batch_size, 1])
        indices = tf.reshape(tf.range(self.batch_size), [self.batch_size, 1])
        concated = tf.concat(axis=1, values=[indices, sparse_labels])
        num_classes = logits[0].get_shape()[-1].value
        dense_labels = tf.sparse_to_dense(concated,
            [self.batch_size, num_classes],
            1.0, 0.0)
        return tf.losses.softmax_cross_entropy(dense_labels, logits), acc_op

    def get_pred_loss(self, logits, labels, reduction='weighted_sum_by_nonzero_weights'):
        sparse_labels = tf.reshape(labels, [self.batch_size, 1])
        indices = tf.reshape(tf.range(self.batch_size), [self.batch_size, 1])
        concated = tf.concat(axis=1, values=[indices, sparse_labels])
        num_classes = logits[0].get_shape()[-1].value
        dense_labels = tf.sparse_to_dense(concated,
            [self.batch_size, num_classes],
            1.0, 0.0)
        return tf.losses.softmax_cross_entropy(dense_labels, logits, reduction=reduction)




    def get_tower_loss_and_acc(self, scope, image_batch, label_batch, tower_name='tower'):
        accuracy_loss, acc_op = self.get_pred_loss_and_acc(image_batch, label_batch)
        l2_loss = self.calculate_l2_loss(self.get_kernel_tensors())
        print('adding l2 loss, l2 factor:', self.l2_factor)
        total_loss = tf.add(accuracy_loss, self.l2_factor * l2_loss, name='total_loss')
        tf.summary.scalar('pred loss', accuracy_loss)
        tf.summary.scalar('l2 loss', l2_loss)
        tf.summary.scalar('total loss', total_loss)
        # tf.losses.add_loss(l2_loss)
        # tf.losses.add_loss(total_loss)
        # losses = tf.losses.get_losses(scope=scope)
        # for l in losses:
        #     loss_name = re.sub('%s_[0-9]*/' % tower_name, '', l.op.name)
        #     tf.summary.scalar(loss_name, l)
        return total_loss, acc_op


    def clear(self):
        self.stop_queue_runners()
        self.sess.close()
        tf.reset_default_graph()
        print('model cleared')

    def load_weights_from_file(self, file_path):
        if file_path is None:
            print('NO WEIGHTS TO LOAD!!!')
            self.np_file = 'NONE'
            if self.need_initialization:
                self.initialize()
        elif file_path.endswith('npy'):
            self.load_weights_from_np(file_path)
        else:
            self.load_weights_from_hdf5(file_path)

    def show_variables(self):
        vs = self.get_global_variables()
        for v in vs:
            print(v.name, v.get_shape())

    def load_weights_from_np(self, np_file):
        ignore_patterns = ['tower_[0-9]/']
        self.np_file = np_file
        if self.need_initialization:
            self.initialize()
        vars = self.get_key_variables()
        _dic = np.load(np_file).item()
        dic = {}
        for k, v in _dic.items():
            dic[eliminate_all_patterns(k, ignore_patterns)] = v
        cnt = 0
        tensors = []
        values = []
        for t in vars:
            name = eliminate_all_patterns(t.name, ignore_patterns)
            if name in dic:
                tensors.append(t)
                values.append(dic[name])
                # print(name, t.get_shape(), dic[name].shape)
                # print(name)
                cnt += 1
            else:
                print('cannot find matched value for variable ', name)
        if tensors:
            self.batch_set_value(tensors, values)
        print('successfully loaded np. {} global variables, {} tensors assigned'.format(len(vars), cnt))

    def _get_dic_of_variables(self, vars):
        result = {}
        values = self.get_value(vars)
        for var, value in zip(vars, values):
            result[var.name] = value
        if self.deps is not None:
            result['deps'] = self.deps
        return result

    def save_weights_to_np(self, np_file):
        result = self._get_dic_of_variables(self.get_key_variables())
        np.save(np_file, result)
        print('save {} key variables to numpy file {}'.format(len(result), np_file))

    def save_weights_to_hdf5(self, hdf5_file):
        result = self._get_dic_of_variables(self.get_key_variables())
        save_hdf5(result, hdf5_file)
        print('save {} key variables to hdf5 file {}'.format(len(result), hdf5_file))


    def save_moving_average_weights_to_hdf5(self, hdf5_file, moving_averages):
        key_variables = self.get_key_variables()
        names = []
        fetches = []
        for kv in key_variables:
            names.append(kv.name)
            mav = moving_averages.average(kv)
            if mav is None:
                fetches.append(kv)
            else:
                fetches.append(mav)
        result = self._get_dic_of_variables(fetches)
        save_hdf5(result, hdf5_file)
        print('save {} moving average key variables to hdf5 file {}'.format(len(result), hdf5_file))


    def load_weights_from_hdf5(self, hdf5_file):
        ignore_patterns = ['tower_[0-9]/', ':0','/ExponentialMovingAverage']
        self.np_file = hdf5_file
        if self.need_initialization:
            self.initialize()
        vars = self.get_key_variables()
        _dic = read_hdf5(hdf5_file)
        dic = {}
        for k, v in _dic.items():
            dic[eliminate_all_patterns(k, ignore_patterns)] = v
        cnt = 0
        tensors = []
        values = []
        for t in vars:
            name = eliminate_all_patterns(t.name, ignore_patterns)
            if name in dic:
                tensors.append(t)
                values.append(dic[name])
                print(name, t.get_shape(), dic[name].shape)
                # print(name)
                cnt += 1
            else:
                print('cannot find matched value for variable ', name)
        if tensors:
            self.batch_set_value(tensors, values)
        print('successfully loaded hdf5. {} global variables, {} tensors assigned'.format(len(vars), cnt))

    def set_value(self, t, value):
        with self.graph.as_default():
            ph = tf.placeholder(t.dtype, t.get_shape())
            op = t.assign(ph)
            self.sess.run(op, feed_dict={ph: value})

    def batch_set_value(self, tensors, values):
        assert len(tensors) == len(values)
        with self.graph.as_default():
            ops = []
            feed_dict = {}
            for t, v in zip(tensors, values):
                if t is not None:
                    ph = tf.placeholder(t.dtype, t.get_shape())
                    ops.append(t.assign(ph))
                    feed_dict[ph] = v
            self.sess.run(ops, feed_dict=feed_dict)

    def get_value(self, t):
        if t is None:
            return None
        if not (type(t) is list or type(t) is tuple):
            return self.sess.run(t)
        fetch_list = []
        none_idxes = []
        for i, itm in enumerate(t):
            if itm is None:
                none_idxes.append(i)
            else:
                fetch_list.append(itm)
        values = self.sess.run(fetch_list)
        result = []
        values_i = 0
        for i in range(len(t)):
            if i in none_idxes:
                result.append(None)
            else:
                result.append(values[values_i])
                values_i += 1
        return result

    def get_constant_lr(self, lr_value, init_step=0):
        self.prepare_global_step(init_step=init_step)
        with self.graph.as_default():
            print('constant lr, values {}'.format(lr_value))
            self.lr = tf.Variable(lr_value, trainable=False, dtype=tf.float32, name='lr')
        return self.lr

    def get_piecewise_lr(self, values, boundaries_steps=None, boundaries_epochs=None, init_step=0):
        if boundaries_steps is None:
            assert boundaries_epochs is not None
            steps_per_epoch = self.dataset.num_examples_per_epoch() // self.batch_size
            boundaries_steps = [float(step * steps_per_epoch) for step in boundaries_epochs]
        else:
            assert boundaries_epochs is None
            boundaries_steps = [float(s) for s in boundaries_steps]
        self.prepare_global_step(init_step=init_step)
        with self.graph.as_default():
            print('piecewise lr, values {}, boundaries {}'.format(values, boundaries_steps))
            self.lr = tf.train.piecewise_constant(self.global_step, boundaries_steps, values)
        return self.lr

    def get_exponential_lr(self, init_lr, lr_decay_factor, steps_per_gpu_per_decay=None, epochs_per_gpu_per_decay=None, init_step=0):
        if steps_per_gpu_per_decay is None:
            assert epochs_per_gpu_per_decay is not None
            steps_per_epoch = self.dataset.num_examples_per_epoch() // self.batch_size
            steps_per_gpu_per_decay = steps_per_epoch * epochs_per_gpu_per_decay
        else:
            assert epochs_per_gpu_per_decay is None
        self.prepare_global_step(init_step=init_step)
        with self.graph.as_default():
            self.lr = tf.train.exponential_decay(init_lr, self.global_step, steps_per_gpu_per_decay, lr_decay_factor, staircase=True)
            print('exponential lr, init lr {}, decay steps {}, decay factor {}'.format(init_lr, steps_per_gpu_per_decay, lr_decay_factor))
        return self.lr

    def prepare_global_step(self, init_step=0):
        with self.graph.as_default():
            self.global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(init_step), trainable=False)
        print('global step variable prepared')
        return self.global_step

    def get_key_variables(self):
        result = self.get_trainable_variables()
        result += self.get_moving_mean_tensors()
        result += self.get_moving_variance_tensors()
        return result

    def set_and_start_queue_runners(self):
        if not self.queue_runner_started:
            self.coord = tf.train.Coordinator()
            self.threads = []
            for qr in self.graph.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                self.threads.extend(qr.create_threads(self.sess, coord=self.coord, daemon=True,
                    start=True))
            self.queue_runner_started = True
        return self.coord

    def stop_queue_runners(self):
        if self.queue_runner_started:
            # self.coord.join(self.threads, stop_grace_period_secs=10)
            self.coord.request_stop()
            print('queue runners stopped')
            self.queue_runner_started = False
        else:
            print('no queue runners to stop')


    def __del__(self):
        self.clear()
