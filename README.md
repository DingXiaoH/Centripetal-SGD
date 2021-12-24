# Centripetal-SGD

2021/01/08: This new version supports pruning with multi-GPU training. Code for pruning the torchvision standard ResNet-50 is released. The old version is moved into the "deprecated" directory. 

This repository contains the codes for the following CVPR-2019 paper 

[Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html).

And the journal extension

[Manipulating Identical Filter Redundancy for Efficient Pruning on Deep and Complicated CNN](https://arxiv.org/abs/2107.14444).

This demo will show you how to prune ResNet-50 on ImageNet with multiple GPUs (Distributed Data parallel) and ResNet-56 on CIFAR-10.

The results reproduced on the standard torchvision version of ResNet-50 (FLOPs=4.09B, top1-accuracy=76.15%) are

| Final width         | FLOPs reduction           | Top-1 accuracy  | Download |
| ------------- |:------------:| -----:| :------------:|
| Original torchvision model	|-|	76.15 |		-|	
| Internal layers 70%   | 36% 	|  	75.94 |		https://drive.google.com/file/d/1kFyc8xH2bRAi-e3v1iC529hTLBIVASGa/view?usp=sharing|
| Internal layers 60%   | 46% 	|  	75.80 |		https://drive.google.com/file/d/1_2tWF-St06KVj49c8yLrAlWUv8fv-LLk/view?usp=sharing|
| Internal layers 50%   | 56% 	|  	75.29 |		https://drive.google.com/file/d/1BndZeq3QkMOAE3wLfltt5SzCJwVF9PLV/view?usp=sharing|

These results are presented in the journal extension, not the CVPR version.

Citation:

	@inproceedings{ding2019centripetal,
  		title={Centripetal sgd for pruning very deep convolutional networks with complicated structure},
  		author={Ding, Xiaohan and Ding, Guiguang and Guo, Yuchen and Han, Jungong},
  		booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  		pages={4943--4953},
  		year={2019}
	}
	
And

	@article{ding2021manipulating,
  		title={Manipulating Identical Filter Redundancy for Efficient Pruning on Deep and Complicated CNN},
  		author={Ding, Xiaohan and Hao, Tianxiang and Han, Jungong and Guo, Yuchen and Ding, Guiguang},
  		journal={arXiv preprint arXiv:2107.14444},
  	year={2021}
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

## Download a pruned model, test, and use it for your own tasks

Download any of the models above, and run like
```
python ndp_test.py sres50 csgd_res50_internal70.hdf5
```
The model can be used for your own tasks like detection and segmentation as usual.

## How to customize the structure of the final network?

For any conv net, the width of every conv layer is defined by an array named "deps". For example, the original deps of ResNet-50 is
```
RESNET50_ORIGIN_DEPS_FLATTENED = [64,256,64,64,256,64,64,256,64,64,256,512,128,128,512,128,128,512,128,128,512,128,128,512,
                                  1024,256, 256, 1024,256, 256, 1024,256, 256, 1024,256, 256, 1024,256, 256, 1024,256, 256, 1024,
                                  2048,512, 512, 2048,512, 512, 2048,512, 512, 2048]
```
Note that we build the projection (1x1 conv shortcut) layer before the parallel residual block (L61 in stagewise_resnet.py), so that its width (256) preceds the widths of the three layers of the residual block (64, 64, 256). In do_csgd.py, "itr_deps" defines the target structure of the pruned model for each iteration. So if you want to customize the final width by pruning every internal layer by 42% and the other troublesome layers by 39%, do something like this
```
final_deps = np.array(RESNET50_ORIGIN_DEPS_FLATTENED)
for i in range(1, len(RESNET50_ORIGIN_DEPS_FLATTENED)):		# starts from 0 if you want to prune the first layer
    if i in RESNET50_INTERNAL_KERNEL_IDXES:
        final_deps[i] = int(0.58 * final_deps[i])
    else:
        final_deps[i] = int(0.61 * final_deps[i])
itr_deps = [final_deps]		# if you want to do it in one iteration. You can define a series of deps to do it in several iterations, like "generate_itr_to_target_deps_by_schedule_vector".
```


## Contact
dxh17@mails.tsinghua.edu.cn

Google Scholar Profile: https://scholar.google.com/citations?user=CIjw0KoAAAAJ&hl=en

My open-sourced papers and repos: 

The **Structural Re-parameterization Universe**:

1. RepMLP (preprint, 2021) **MLP-style building block and Architecture**\
[RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality](https://arxiv.org/abs/2112.11081)\
[code](https://github.com/DingXiaoH/RepMLP).

2. RepVGG (CVPR 2021) **A super simple and powerful VGG-style ConvNet architecture**. Up to **84.16%** ImageNet top-1 accuracy!\
[RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)\
[code](https://github.com/DingXiaoH/RepVGG).

3. ResRep (ICCV 2021) **State-of-the-art** channel pruning (Res50, 55\% FLOPs reduction, 76.15\% acc)\
[ResRep: Lossless CNN Pruning via Decoupling Remembering and Forgetting](https://openaccess.thecvf.com/content/ICCV2021/papers/Ding_ResRep_Lossless_CNN_Pruning_via_Decoupling_Remembering_and_Forgetting_ICCV_2021_paper.pdf)\
[code](https://github.com/DingXiaoH/ResRep).

4. ACB (ICCV 2019) is a CNN component without any inference-time costs. The first work of our Structural Re-parameterization Universe.\
[ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf).\
[code](https://github.com/DingXiaoH/ACNet). 

5. DBB (CVPR 2021) is a CNN component with higher performance than ACB and still no inference-time costs. Sometimes I call it ACNet v2 because "DBB" is 2 bits larger than "ACB" in ASCII (lol).\
[Diverse Branch Block: Building a Convolution as an Inception-like Unit](https://arxiv.org/abs/2103.13425)\
[code](https://github.com/DingXiaoH/DiverseBranchBlock).

6. COMING SOON

7. COMING SOON

**Model compression and acceleration**:

1. (CVPR 2019) Channel pruning: [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html)\
[code](https://github.com/DingXiaoH/Centripetal-SGD)

2. (ICML 2019) Channel pruning: [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](http://proceedings.mlr.press/v97/ding19a.html)\
[code](https://github.com/DingXiaoH/AOFP)

3. (NeurIPS 2019) Unstructured pruning: [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](http://papers.nips.cc/paper/8867-global-sparse-momentum-sgd-for-pruning-very-deep-neural-networks.pdf)\
[code](https://github.com/DingXiaoH/GSM-SGD)
