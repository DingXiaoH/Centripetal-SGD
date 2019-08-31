# Centripetal-SGD

This repository contains the codes for the following CVPR-2019 paper 

[Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html).

The codes are based on Tensorflow 1.11.

I will re-implement this using pytorch after finishing my pytorch experiment platform. Maybe in a month.

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


## Example Usage
  
This repo holds the example code for pruning DenseNet-40 on CIFAR-10. 

1. Install Tensorflow-gpu-1.11

2. Prepare the CIFAR-10 dataset in tfrecord format. Please follow https://github.com/tensorflow/models/blob/master/research/slim/datasets/download_and_convert_cifar10.py, download the CIFAR-10 dataset, convert it to tfrecord format, rename the two output files as train.tfrecords and validation.tfrecords, and modify the value of DATA_PATH in tf_dataset.py.

3. Prune a DenseNet-40 to 3 filters per layer based on the magnitude of kernels and finetune it. Then evaluate the model.

```
python csgd_standalone.py magnitude1
python csgd_standalone.py eval magnitude1_trained.hdf5
```

4. Train a DenseNet-40 using C-SGD and trim it to obtain the same final structure. Then evaluate the model.

```
python csgd_standalone.py csgd1
python csgd_standalone.py eval dc40_csgd1_itr0_prunedweights.hdf5
```

## Contact
dxh17@mails.tsinghua.edu.cn

Google Scholar Profile: https://scholar.google.com/citations?user=CIjw0KoAAAAJ&hl=en
