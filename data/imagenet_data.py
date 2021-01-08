import torch
import torchvision.transforms as transforms
from data.dataset_util import DataIterator
import os
import torchvision.datasets as datasets

IMGNET_TRAIN_DIR = 'imagenet_data'

class ImgnetStdTrainData(object):

    def __init__(self, distributed, batch_size_per_gpu, img_size=224):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        trans = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        self.train_dataset = datasets.ImageFolder(
            os.path.join(IMGNET_TRAIN_DIR, 'train'),
            trans
        )

        if distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
            shuffle = False
        else:
            self.train_sampler = None
            shuffle = True
        self.train_loader = torch.utils.data.DataLoader(
                            self.train_dataset, batch_size=batch_size_per_gpu, sampler=self.train_sampler, shuffle=shuffle,
                            num_workers=4, pin_memory=True, drop_last=True)
        self.dataprovider = DataIterator(self.train_loader)


class ImgnetStdValData(object):
    def __init__(self, batch_size, img_size=224):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize
        ]
        )

        self.val_dataset = datasets.ImageFolder(
            os.path.join(IMGNET_TRAIN_DIR, 'val'),
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
            )
        )

        self.val_loader = torch.utils.data.DataLoader(
                     self.val_dataset, batch_size=batch_size, shuffle=False,
                     num_workers=4, pin_memory=True
                    )
        self.dataprovider = DataIterator(self.val_loader)