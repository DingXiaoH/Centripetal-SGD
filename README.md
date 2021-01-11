# Centripetal-SGD

2021/01/08: This new version supports pruning with multi-GPU training. Code for pruning the torchvision standard ResNet-50 is released. The old version is moved into the "deprecated" directory. 

This repository contains the codes for the following CVPR-2019 paper 

[Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html).

This demo will show you how to prune ResNet-50 on ImageNet with multiple GPUs (Distributed Data parallel) and ResNet-56 on CIFAR-10.

The results reproduced on the torchvision version of ResNet-50 (FLOPs=4.09B, top1-accuracy=76.15%) are

| Final width         | FLOPs reduction           | Top-1 accuracy  | Download |
| ------------- |:------------:| -----:|
| Original torchvision model	|-|	76.15 |		-|	
| Internal layers 70%   | 36% 	|  	75.94 |		https://drive.google.com/file/d/1kFyc8xH2bRAi-e3v1iC529hTLBIVASGa/view?usp=sharing|
| Internal layers 60%   | 46% 	|  	75.80 |		https://drive.google.com/file/d/1_2tWF-St06KVj49c8yLrAlWUv8fv-LLk/view?usp=sharing|
| Internal layers 50%   | 56% 	|  	75.80 |		https://drive.google.com/file/d/1_2tWF-St06KVj49c8yLrAlWUv8fv-LLk/view?usp=sharing|

Citation:

	@inproceedings{ding2019centripetal,
  		title={Centripetal sgd for pruning very deep convolutional networks with complicated structure},
  		author={Ding, Xiaohan and Ding, Guiguang and Guo, Yuchen and Han, Jungong},
  		booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  		pages={4943--4953},
  		year={2019}
	}

## Introduction

Filter pruning, a.k.a. network slimming or channel pruning, aims to remove some filters from a CNN so as to slim it with acceptable performance drop. We seek to make some filters increasingly close and eventually identical for network slimming. To this end, we propose Centripetal SGD (C-SGD), a novel optimization method, which can train several filters to collapse into a single point in the parameter hyperspace. When the training is completed, the removal of the identical filters can trim the network with NO performance
loss, thus no finetuning is needed. By doing so, we have partly solved an open problem of constrained filter pruning on CNNs with complicated structure, where some layers must be pruned following others.


## PyTorch Example Usage: Pruning ResNet-50 with multiple GPUs.

1. Enter this directory.

2. Make a soft link to your ImageNet directory, which contains "train" and "val" directories.
```
ln -s YOUR_PATH_TO_IMAGENET imagenet_data
```

3. Set the environment variables.
```
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

4. Download the official torchvision model, rename the parameters in our namestyle, and save the weights to "torchvision_res50.hdf5".
```
python transform_torchvision.py
```

5. Run Centripetal SGD to prune the internal layers of ResNet-50 to 70% of the original width, then 60%, then 50%, 40%, 30%.
```
python -m torch.distributed.launch --nproc_per_node=8 csgd/do_csgd.py -a sres50 -i 0
python -m torch.distributed.launch --nproc_per_node=8 csgd/do_csgd.py -a sres50 -i 1
python -m torch.distributed.launch --nproc_per_node=8 csgd/do_csgd.py -a sres50 -i 2
python -m torch.distributed.launch --nproc_per_node=8 csgd/do_csgd.py -a sres50 -i 3
python -m torch.distributed.launch --nproc_per_node=8 csgd/do_csgd.py -a sres50 -i 4
```


## PyTorch Example Usage: Pruning ResNet-56 on CIFAR-10

We train a ResNet-56 (with 16-32-64 channels) and iteratively slim it into 13/16, 11/16 and 5/8 of the original width.

1. Enter this directory.

2. Make a soft link to your CIFAR-10 directory. If the dataset is not found in the directory, it will be automatically downloaded.
```
ln -s YOUR_PATH_TO_CIFAR cifar10_data
```

3. Set the environment variables.
```
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
```

4. Run Centripetal SGD to train a base ResNet-56, then globally slim it into 13/16, 11/16, 5/8 of the original width.
```
python csgd/do_csgd.py -a src56 -i 0
python csgd/do_csgd.py -a src56 -i 1
python csgd/do_csgd.py -a src56 -i 2
```

## How to customize the structure of the final network?

For any conv net, the width of every conv layer is defined by deps.

## Contact
dxh17@mails.tsinghua.edu.cn

Google Scholar Profile: https://scholar.google.com/citations?user=CIjw0KoAAAAJ&hl=en

My open-sourced papers and repos: 

**State-of-the-art** channel pruning (preprint, 2020): [Lossless CNN Channel Pruning via Gradient Resetting and Convolutional Re-parameterization](https://arxiv.org/abs/2007.03260) (https://github.com/DingXiaoH/ResRep)

CNN component (ICCV 2019): [ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf) (https://github.com/DingXiaoH/ACNet)

Channel pruning (CVPR 2019): [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html) (https://github.com/DingXiaoH/Centripetal-SGD)

Channel pruning (ICML 2019): [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](http://proceedings.mlr.press/v97/ding19a.html) (https://github.com/DingXiaoH/AOFP)

Unstructured pruning (NeurIPS 2019): [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](http://papers.nips.cc/paper/8867-global-sparse-momentum-sgd-for-pruning-very-deep-neural-networks.pdf) (https://github.com/DingXiaoH/GSM-SGD)
