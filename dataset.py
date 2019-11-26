import torch
from torchvision import datasets, transforms

MNIST_PATH = '/home/dingxiaohan/datasets/torch_mnist/'
CIFAR10_PATH = '/home/dingxiaohan/datasets/cifar-10-batches-py/'
CH_PATH = '/home/dingxiaohan/datasets/torch_ch/'
SVHN_PATH = '/home/dingxiaohan/datasets/torch_svhn/'



def load_cuda_data(data_loader, dataset_name):
    if dataset_name == 'imagenet':
        data_dict = next(data_loader)
        data = data_dict['data']
        label = data_dict['label']
        data = torch.from_numpy(data).cuda()
        label = torch.from_numpy(label).type(torch.long).cuda()
    else:
        data, label = next(data_loader)
        data = data.cuda()
        label = label.cuda()
    return data, label

class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


def create_dataset(dataset_name, subset, batch_size):
    assert dataset_name in ['imagenet', 'cifar10', 'ch', 'svhn', 'mnist']
    assert subset in ['train', 'val']
    if dataset_name == 'imagenet':
        from ntools.megtools.classification.config import DpflowProviderMaker, DataproProviderMaker
        if subset == 'train':
            with open('imagenet_train_conn.txt', 'r') as f:
                conn = f.readline().strip()
            return DpflowProviderMaker(conn=conn,
                                     entry_names=['image', 'label'],
                                     output_names=['data', 'label'],
                                     descriptor={'data': {'shape': [batch_size, 3, 224, 224]}, 'label': {'shape': [batch_size]}},
                                     buffer_size=16,
                                     group_id = None,
                                     enable_multiprocessing=False)()
        else:
            return DataproProviderMaker(config_file='provider_config_val.txt',
                                          provider_name='provider_cfg_val',
                                          entry_names=['image_val', 'label'],
                                          output_names=['data', 'label'])()

    #   copied from https://github.com/pytorch/examples/blob/master/mnist/main.py
    elif dataset_name == 'mnist':
        if subset == 'train':
            return InfiniteDataLoader(datasets.MNIST(MNIST_PATH, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size, shuffle=True)
        else:
            return InfiniteDataLoader(datasets.MNIST(MNIST_PATH, train=False, transform=transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                batch_size=batch_size, shuffle=False)



    elif dataset_name == 'cifar10':
        if subset == 'train':
            return InfiniteDataLoader(datasets.CIFAR10(CIFAR10_PATH, train=True, download=False,
                               transform=transforms.Compose([
                                   transforms.Pad(padding=(4, 4, 4, 4)),
                                   transforms.RandomCrop(32),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                                batch_size=batch_size, shuffle=True)
        else:
            return InfiniteDataLoader(datasets.CIFAR10(CIFAR10_PATH, train=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                                batch_size=batch_size, shuffle=False)

    elif dataset_name == 'ch':
        if subset == 'train':
            return InfiniteDataLoader(datasets.CIFAR100(CH_PATH, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.Pad(padding=(4, 4, 4, 4)),
                                   transforms.RandomCrop(32),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                                batch_size=batch_size, shuffle=True)
        else:
            return InfiniteDataLoader(datasets.CIFAR100(CH_PATH, train=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                                batch_size=batch_size, shuffle=False)

    else:
        assert False


def num_train_examples_per_epoch(dataset_name):
    if dataset_name == 'imagenet':
        return 1281167
    elif dataset_name == 'mnist':
        return 60000
    elif dataset_name in ['cifar10', 'ch']:
        return 50000
    else:
        assert False

def num_iters_per_epoch(cfg):
    return num_train_examples_per_epoch(cfg.dataset_name) // cfg.global_batch_size