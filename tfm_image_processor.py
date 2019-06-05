
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from abc import abstractmethod

CIFAR10_MEAN = [125.30690002, 122.95014954, 113.86599731]  #not RGB! already switched!
CIFAR10_MODES = ['train', 'eval']

INPUT_QUEUE_MEMORY_FACTOR=16


def normalized_image(images):
    # Rescale from [0, 255] to [0, 2]
    images = tf.multiply(images, 1. / 127.5)
    # Rescale to [-1, 1]
    return tf.subtract(images, 1.0)

class ImagePreprocessor(object):

    def __init__(self, mode, dataset, image_size, batch_size, num_preprocess_threads=4, num_readers=None):
        self.mode = mode
        self.dataset = dataset
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_preprocess_threads = num_preprocess_threads
        self.num_readers = num_readers

    @abstractmethod
    def get_batch_input_tensors(self):
        pass

    @abstractmethod
    def preprocess_fn(self, image_size, image_buffer, bbox, mode, thread_id=0):
        pass



class CIFAR10Preprocessor(ImagePreprocessor):

    def __init__(self, mode, dataset, image_size, batch_size, num_preprocess_threads=4, num_readers=None, need_shuffle=False):
        super(CIFAR10Preprocessor, self).__init__(mode, dataset, image_size, batch_size, num_preprocess_threads, num_readers)
        assert self.mode in CIFAR10_MODES
        self.need_shuffle = need_shuffle
        if self.num_readers is None:
            if mode == 'eval':
                self.num_readers = 1
            else:
                self.num_readers = 4

    def get_batch_input_tensors(self):
        with tf.name_scope('batch_processing'):
            data_files = self.dataset.data_files()
            if data_files is None:
                raise ValueError('No data files found for this dataset')

            # Create filename_queue
            if self.mode == 'train' or self.need_shuffle:
                filename_queue = tf.train.string_input_producer(data_files,
                                                                shuffle=True,
                                                                capacity=16)
            else:
                filename_queue = tf.train.string_input_producer(data_files,
                                                                shuffle=False,
                                                                capacity=1)

            if self.num_preprocess_threads % 2:
                raise ValueError('Please make num_preprocess_threads a multiple '
                                 'of 2 (%d % 2 != 0).', self.num_preprocess_threads)

            if self.num_readers < 1:
                raise ValueError('Please make num_readers at least 1')

            # Approximate number of examples per shard.
            examples_per_shard = 1024
            # Size the random shuffle queue to balance between good global
            # mixing (more examples) and memory use (fewer examples).
            # 1 image uses 299*299*3*4 bytes = 1MB
            # The default input_queue_memory_factor is 16 implying a shuffling queue
            # size: examples_per_shard * 16 * 1MB = 17.6GB
            min_queue_examples = examples_per_shard * INPUT_QUEUE_MEMORY_FACTOR
            if self.mode == 'train' or self.need_shuffle:
                print('use random shuffle example queue')
                examples_queue = tf.RandomShuffleQueue(
                    capacity=min_queue_examples + 3 * self.batch_size,
                    min_after_dequeue=min_queue_examples,
                    dtypes=[tf.string])
            else:
                print('use FIFO example queue')
                examples_queue = tf.FIFOQueue(
                    capacity=examples_per_shard + 3 * self.batch_size,
                    dtypes=[tf.string])

            # Create multiple readers to populate the queue of examples.
            if self.num_readers > 1:
                enqueue_ops = []
                for _ in range(self.num_readers):
                    reader = self.dataset.reader()
                    _, value = reader.read(filename_queue)
                    enqueue_ops.append(examples_queue.enqueue([value]))

                tf.train.queue_runner.add_queue_runner(
                    tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
                example_serialized = examples_queue.dequeue()
            else:
                reader = self.dataset.reader()
                _, example_serialized = reader.read(filename_queue)

            images_and_labels = []
            for thread_id in range(self.num_preprocess_threads):
                # Parse a serialized Example proto to extract the image and metadata.
                image, label_index = self.cifar10_parse_example_proto(example_serialized)
                # image = self.cifar10_preprocess(image_buffer)
                images_and_labels.append([image, label_index])

            images, label_index_batch = tf.train.batch_join(
                images_and_labels,
                batch_size=self.batch_size,
                capacity=2 * self.num_preprocess_threads * self.batch_size)

            # Reshape images into these desired dimensions.
            height = self.image_size
            width = self.image_size
            depth = 3

            images = tf.cast(images, tf.float32)
            images = tf.reshape(images, shape=[self.batch_size, height, width, depth])

            # Display the training images in the visualizer.
            tf.summary.image('images', images)
            return images, tf.reshape(label_index_batch, [self.batch_size])


    def cifar10_distort(self, image):
        image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
        image = tf.random_crop(image, [self.image_size, self.image_size, 3])
        image = tf.image.random_flip_left_right(image)
        return image


    def cifar10_parse_example_proto(self, example_serialized):
        feature_map = {
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                default_value=''),
            'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                default_value=-1)
        }
        features = tf.parse_single_example(example_serialized, feature_map)
        image_buffer = features['image/encoded']
        label = tf.cast(features['image/class/label'], dtype=tf.int32)

        image = tf.image.decode_jpeg(image_buffer, channels=3, dct_method='INTEGER_FAST')
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_image_with_crop_or_pad(image, 32, 32)

        # CHW -> HWC
        # image = tf.cast(tf.transpose(tf.reshape(image, [3, self.image_size, self.image_size]), [1, 2, 0]), tf.float32)

        # red, green, blue = tf.split(value=image, num_or_size_splits=[1, 1, 1], axis=2)
        # image = tf.concat(values=[
        #     blue - CIFAR10_MEAN[0],
        #     green - CIFAR10_MEAN[1],
        #     red - CIFAR10_MEAN[2],
        # ], axis=2)

        if self.mode == 'train':
            print('distort image')
            image = self.cifar10_distort(image)

        # image = tf.subtract(image, tf.constant(np.array(CIFAR10_MEAN, dtype=np.float32), dtype=tf.float32))
        image = normalized_image(image)
        return image, label