OVERALL_EVAL_RECORD_FILE = 'overall_eval_records.txt'
from collections import namedtuple

LRSchedule = namedtuple('LRSchedule', ['base_lr', 'max_epochs', 'lr_epoch_boundaries', 'lr_decay_factor',
                                       'linear_final_lr'])

import numpy as np



def parse_usual_lr_schedule(try_arg, keyword='lrs{}'):
    if keyword.format(1) in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=500, lr_epoch_boundaries=[100, 200, 300, 400], lr_decay_factor=0.3,
                         linear_final_lr=None)
    elif keyword.format(2) in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=500, lr_epoch_boundaries=[100, 200, 300, 400], lr_decay_factor=0.1,
                         linear_final_lr=None)
    elif keyword.format(3) in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=800, lr_epoch_boundaries=[200, 400, 600], lr_decay_factor=0.1,
                         linear_final_lr=None)
    elif keyword.format(4) in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=80, lr_epoch_boundaries=[20, 40, 60], lr_decay_factor=0.1,
                         linear_final_lr=None)
    elif keyword.format(5) in try_arg:
        lrs = LRSchedule(base_lr=0.05, max_epochs=200, lr_epoch_boundaries=[50, 100, 150], lr_decay_factor=0.1,
                         linear_final_lr=None)
    elif keyword.format(6) in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=360, lr_epoch_boundaries=[90, 180, 240, 300], lr_decay_factor=0.2,
                         linear_final_lr=None)
    elif keyword.format(7) in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=800, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=1e-4)
    elif keyword.format(8) in try_arg:  # may be enough for MobileNet v1 on CIFARs
        lrs = LRSchedule(base_lr=0.1, max_epochs=400, lr_epoch_boundaries=[100, 200, 300], lr_decay_factor=0.1,
                         linear_final_lr=None)
    elif keyword.format(9) in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=200, lr_epoch_boundaries=[50, 100, 150], lr_decay_factor=0.1,
                         linear_final_lr=None)

    elif keyword.format('A') in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=100, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=1e-5)
    elif keyword.format('B') in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=100, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=1e-6)
    elif keyword.format('C') in try_arg:
        lrs = LRSchedule(base_lr=0.2, max_epochs=125, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=0)
    elif keyword.format('D') in try_arg:
        lrs = LRSchedule(base_lr=0.001, max_epochs=20, lr_epoch_boundaries=[5, 10], lr_decay_factor=0.1,
                         linear_final_lr=None)
    elif keyword.format('E') in try_arg:
        lrs = LRSchedule(base_lr=0.001, max_epochs=300, lr_epoch_boundaries=[100, 200], lr_decay_factor=0.1,
                         linear_final_lr=None)

    elif keyword.format('F') in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=120, lr_epoch_boundaries=[30, 60, 90, 110], lr_decay_factor=0.1,
                         linear_final_lr=None)
    #   for VGG and CFQKBN
    elif keyword.format('G') in try_arg:
        lrs = LRSchedule(base_lr=0.05, max_epochs=800, lr_epoch_boundaries=[200, 400, 600], lr_decay_factor=0.1,
                         linear_final_lr=None)
    elif keyword.format('H') in try_arg:
        lrs = LRSchedule(base_lr=0.025, max_epochs=200, lr_epoch_boundaries=[50, 100, 150], lr_decay_factor=0.1,
                         linear_final_lr=None)

    elif keyword.format('L') in try_arg:
        lrs = LRSchedule(base_lr=0.1, max_epochs=1600, lr_epoch_boundaries=[400, 800, 1200], lr_decay_factor=0.1,
                         linear_final_lr=None)



    elif keyword.format('X') in try_arg:
        lrs = LRSchedule(base_lr=0.2, max_epochs=6, lr_epoch_boundaries=None, lr_decay_factor=None,
                         linear_final_lr=0)

    elif keyword.replace('{}', '') in try_arg:
        raise ValueError('Unsupported lrs config.')
    else:
        lrs = None
    return lrs



MASK_VALUE_KEYWORD = 'base_mask_value'

SIMPLE_ALEXNET_DEPS = np.array([64, 192, 384, 384, 256])

MOBILENET_V2_100_DEPS = [32, 32, 16, 96, 96, 24, 144, 144, 24, 144, 144, 32, 192, 192, 32, 192, 192, 32, 192, 192, 64, 384, 384, 64, 384, 384, 64, 384, 384, 64, 384, 384, 96, 576, 576, 96, 576, 576, 96, 576, 576, 160, 960, 960, 160, 960, 960, 160, 960, 960, 320, 1280, 1001]

MOBILENET_V2_EXPAND_IDS =       [3, 6,   9, 12, 15,    18, 21, 24, 27,   30, 33, 36,   39, 42, 45, 48]
MOBILENET_V2_DEPTHWISE_IDS = [1, 4, 7,   10, 13, 16,   19, 22, 25, 28,   31, 34, 37,   40, 43, 46, 49]
MOBILENET_V2_PROJECT_IDS =   [2, 5, 8,   11, 14, 17,   20, 23, 26, 29,   32, 35, 38,   41, 44, 47, 50]

#   20: expanded_conv_6
#   32: expanded_conv_10
#   41: expanded_conv_13
VGG_ORIGIN_DEPS = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]

CFQK_ORIGIN_DEPS = np.array([32, 32, 64], dtype=np.int32)

MOBILENET_V2_succeeding_STRATEGY = 'simple'

MOBILENET_V2_FOLLOW_DICT = {8:5,
                            14:11, 17:11,
                            23:20, 26:20, 29:20,
                            35:32, 38:32,
                            44:41, 47:41}
MOBILENET_V2_FOLLOW_DICT.update({i:(i-1) for i in MOBILENET_V2_DEPTHWISE_IDS})


MOBILENET_V2_075_DEPS = [24, 24, 16, 96, 96, 24, 144, 144, 24, 144, 144, 24, 144, 144, 24, 144, 144, 24, 144, 144, 48, 288, 288, 48, 288, 288, 48, 288, 288, 48, 288, 288, 72, 432, 432, 72, 432, 432, 72, 432, 432, 120, 720, 720, 120, 720, 720, 120, 720, 720, 240, 1280, 1001]

MOBILENET_V2_050_DEPS = [16, 16, 8, 48, 48, 16, 96, 96, 16, 96, 96, 16, 96, 96, 16, 96, 96, 16, 96, 96, 32, 192, 192, 32, 192, 192, 32, 192, 192, 32, 192, 192, 48, 288, 288, 48, 288, 288, 48, 288, 288, 80, 480, 480, 80, 480, 480, 80, 480, 480, 160, 1280, 1001]

MOBILENET_V2_035_DEPS = [16, 16, 8, 48, 48, 8, 48, 48, 8, 48, 48, 16, 96, 96, 16, 96, 96, 16, 96, 96, 24, 144, 144, 24, 144, 144, 24, 144, 144, 24, 144, 144, 32, 192, 192, 32, 192, 192, 32, 192, 192, 56, 336, 336, 56, 336, 336, 56, 336, 336, 112, 1280, 1001]

MOBILENET_ITR_TO_TARGET_DEPS = [MOBILENET_V2_075_DEPS, MOBILENET_V2_050_DEPS, MOBILENET_V2_035_DEPS]

MOBILENET_V2_ALTERABLE_LAYERS = range(0, len([16, 16, 8, 48, 48, 8, 48, 48, 8, 48, 48, 16, 96, 96, 16, 96, 96, 16, 96, 96, 24, 144, 144, 24, 144, 144, 24, 144, 144, 24, 144, 144, 32, 192, 192, 32, 192, 192, 32, 192, 192, 56, 336, 336, 56, 336, 336, 56, 336, 336, 112]))

MOBILENET_V2_EQCLS_LAYERS = [i for i in MOBILENET_V2_ALTERABLE_LAYERS if i not in MOBILENET_V2_FOLLOW_DICT]




def wrn_origin_deps_flattened(n, k):
    assert n in [2, 4, 6]   # total_depth = 6n + 4
    filters_in_each_stage = n * 2 + 1
    stage0 = [16]
    stage1 = [16 * k] * filters_in_each_stage
    stage2 = [32 * k] * filters_in_each_stage
    stage3 = [64 * k] * filters_in_each_stage
    return np.array(stage0 + stage1 + stage2 + stage3)

def wrn_pacesetter_idxes(n):
    assert n in [2, 4, 6]
    filters_in_each_stage = n * 2 + 1
    pacesetters = [1, int(filters_in_each_stage)+1, int(2 * filters_in_each_stage)+1]   #[1, 10, 19] for WRN-28-x, for example
    return pacesetters

def wrn_convert_flattened_deps(flattened):
    assert len(flattened) in [16, 28, 40]
    n = int((len(flattened) - 4) // 6)
    assert n in [2, 4, 6]
    pacesetters = wrn_pacesetter_idxes(n)
    result = [flattened[0]]
    for ps in pacesetters:
        assert flattened[ps] == flattened[ps+2]
        stage_deps = []
        for i in range(n):
            stage_deps.append([flattened[ps + 1 + 2 * i], flattened[ps + 2 + 2 * i]])
        result.append(stage_deps)
    return result

##################### general Resnet on CIFAR-10
def rc_origin_deps_flattened(n):
    assert n in [3, 9, 12, 18, 27, 200]
    filters_in_each_stage = n * 2 + 1
    stage1 = [16] * filters_in_each_stage
    stage2 = [32] * filters_in_each_stage
    stage3 = [64] * filters_in_each_stage
    return np.array(stage1 + stage2 + stage3)


def rc_convert_flattened_deps(flattened):
    filters_in_each_stage = len(flattened) / 3
    n = int((filters_in_each_stage - 1) // 2)
    assert n in [3, 9, 12, 18, 27, 200]
    pacesetters = rc_pacesetter_idxes(n)
    result = [flattened[0]]
    for ps in pacesetters:
        assert flattened[ps] == flattened[ps+2]
        stage_deps = []
        for i in range(n):
            stage_deps.append([flattened[ps + 1 + 2 * i], flattened[ps + 2 + 2 * i]])
        result.append(stage_deps)
    return result

def rc_pacesetter_idxes(n):
    assert n in [3, 9, 12, 18, 27, 200]
    filters_in_each_stage = n * 2 + 1
    pacesetters = [0, int(filters_in_each_stage), int(2 * filters_in_each_stage)]
    return pacesetters

def rc_internal_layers(n):
    assert n in [3, 9, 12, 18, 27, 200]
    pacesetters = rc_pacesetter_idxes(n)
    result = []
    for ps in pacesetters:
        for i in range(n):
            result.append(ps + 1 + 2 * i)
    return result

def rc_all_survey_layers(n):
    return rc_pacesetter_idxes(n) + rc_internal_layers(n)

def rc_all_cov_layers(n):
    return range(0, 6*n+3)

def rc_pacesetter_dict(n):
    assert n in [3, 9, 12, 18, 27, 200]
    pacesetters = rc_pacesetter_idxes(n)
    result = {}
    for ps in pacesetters:
        for i in range(0, n+1):
            result[ps + 2 * i] = ps
    return result

def rc_succeeding_strategy(n):
    assert n in [3, 9, 12, 18, 27, 200]
    internal_layers = rc_internal_layers(n)
    result = {i : (i+1) for i in internal_layers}
    result[0] = 1
    follow_dic = rc_pacesetter_dict(n)
    pacesetters = rc_pacesetter_idxes(n)
    layer_before_pacesetters = [i-1 for i in pacesetters]
    for i in follow_dic.keys():
        if i in layer_before_pacesetters:
            result[i] = [i+1, i+2]
        elif i not in pacesetters:
            result[i] = i + 1
    return result

def rc_fc_layer_idx(n):
    assert n in [9, 12, 18, 27, 200]
    return 6*n+3

def rc_stage_to_pacesetter_idx(n):
    ps_ids = rc_pacesetter_idxes(n)
    return {2:ps_ids[0], 3:ps_ids[1], 4:ps_ids[2]}

def rc_flattened_deps_by_stage(rc_n, stage2, stage3, stage4):
    result = rc_origin_deps_flattened(rc_n)
    stage2_ids = (result == 16)
    stage3_ids = (result == 32)
    stage4_ids = (result == 64)
    result[stage2_ids] = stage2
    result[stage3_ids] = stage3
    result[stage4_ids] = stage4
    return result


def convert_flattened_resnet50_deps(deps):
    assert len(deps) == 53
    assert deps[1] == deps[4] and deps[11] == deps[14] and deps[24] == deps[27] and deps[43] == deps[46]
    d = [deps[0]]
    tmp = []
    for i in range(3):
        tmp.append([deps[2 + i * 3], deps[3 + i * 3], deps[4 + i * 3]])
    d.append(tmp)
    tmp = []
    for i in range(4):
        tmp.append([deps[12 + i * 3], deps[13 + i * 3], deps[14 + i * 3]])
    d.append(tmp)
    tmp = []
    for i in range(6):
        tmp.append([deps[25 + i * 3], deps[26 + i * 3], deps[27 + i * 3]])
    d.append(tmp)
    tmp = []
    for i in range(3):
        tmp.append([deps[44 + i * 3], deps[45 + i * 3], deps[46 + i * 3]])
    d.append(tmp)
    return d

def rc_internal_scaled_flattened_deps(rc_n, scale_factor):
    result = np.array(rc_origin_deps_flattened(rc_n))
    for i in rc_internal_layers(rc_n):
        result[i] = np.ceil(scale_factor * result[i])
    return result

RESNET50_ORIGIN_DEPS=[64,[[64,64,256]]*3,
                       [[128,128,512]]*4,
                       [[256, 256, 1024]]*6,
                       [[512, 512, 2048]]*3]
RESNET50_ORIGIN_DEPS_FLATTENED = [64,256,64,64,256,64,64,256,64,64,256,512,128,128,512,128,128,512,128,128,512,128,128,512,
                                  1024,256, 256, 1024,256, 256, 1024,256, 256, 1024,256, 256, 1024,256, 256, 1024,256, 256, 1024,
                                  2048,512, 512, 2048,512, 512, 2048,512, 512, 2048]
RESNET50_ALL_CONV_LAYERS = range(0, len(RESNET50_ORIGIN_DEPS_FLATTENED))
RESNET50_INTERNAL_KERNEL_IDXES = [2,3,5,6,8,9,12,13,15,16,18,19,21,22,25,26,28,29,31,32,34,35,
                                  37,38,40,41,44,45,47,48,50,51]
RESNET50_PACESETTER_IDXES = [1, 11, 24, 43]
RESNET50_ALL_SURVEY_LAYERS = [0] + RESNET50_INTERNAL_KERNEL_IDXES + RESNET50_PACESETTER_IDXES
RESNET50_FOLLOW_DICT = {1:1, 4:1, 7:1, 10:1, 11:11, 14:11, 17:11, 20:11, 23:11, 24:24, 27:24, 30:24, 33:24, 36:24, 39:24, 42:24, 43:43, 46:43, 49:43, 52:43}
# RESNET50_FOLLOWER_DICT = {1:[1,4,7,10], 11:[11,14,17,20,23], 24:[24,27,30,33,36,39,42], 43:[43,46,49,52]}
RESNET50_succeeding_STRATEGY = {i : (i+1) for i in RESNET50_INTERNAL_KERNEL_IDXES}
RESNET50_succeeding_STRATEGY[0] = [1,2]
idxes_before_pacesetters = [i-1 for i in RESNET50_PACESETTER_IDXES]
for i in RESNET50_FOLLOW_DICT.keys():
    if i not in RESNET50_PACESETTER_IDXES:
        if i in idxes_before_pacesetters:
            RESNET50_succeeding_STRATEGY[i] = [i+1, i+2]
        else:
            RESNET50_succeeding_STRATEGY[i] = i+1



resnet_n_to_num_blocks = {
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3)
}

def resnet_bottleneck_origin_deps_converted(res_n):
    num_blocks = resnet_n_to_num_blocks[res_n]
    return [64,[[64,64,256]]*num_blocks[0],
                       [[128,128,512]]*num_blocks[1],
                       [[256, 256, 1024]]*num_blocks[2],
                       [[512, 512, 2048]]*num_blocks[3]]

def _resnet_bottlenck_first_internal_layer_idx_of_stage(num_blocks):
    return [2, 3+num_blocks[0]*3, 4+(num_blocks[0]+num_blocks[1])*3, 5+(num_blocks[0]+num_blocks[1]+num_blocks[2])*3]


def convert_resnet_bottleneck_deps(deps):
    assert len(deps) in [53, 104, 155]
    res_n = len(deps) - 3
    print('converting the flattened deps of resnet-{}'.format(res_n))
    num_blocks = resnet_n_to_num_blocks[res_n]
    #   the idx of the first layer of the stage (not the proj layer)
    start_layer_idx_of_stage = _resnet_bottlenck_first_internal_layer_idx_of_stage(num_blocks)
    d = [deps[0]]
    for stage_idx in range(4):
        tmp = []
        assert deps[start_layer_idx_of_stage[stage_idx] - 1] == deps[start_layer_idx_of_stage[stage_idx] + 2]  # check the proj layer deps
        for i in range(num_blocks[stage_idx]):
            tmp.append([deps[start_layer_idx_of_stage[stage_idx] + i * 3],
                        deps[start_layer_idx_of_stage[stage_idx] + 1 + i * 3], deps[start_layer_idx_of_stage[stage_idx] + 2 + i * 3]])
        d.append(tmp)
    print('converting completed')
    return d

def resnet_bottleneck_origin_deps_flattened(res_n):
    origin_deps_converted = resnet_bottleneck_origin_deps_converted(res_n)
    flattened = [origin_deps_converted[0]]
    for stage_idx in range(4):
        flattened.append(origin_deps_converted[stage_idx+1][0][2])
        for block in origin_deps_converted[stage_idx+1]:
            flattened += block
    return flattened

def resnet_bottleneck_internal_kernel_indices(res_n):
    internals = []
    num_blocks = resnet_n_to_num_blocks[res_n]
    start_layer_idx_of_stage = _resnet_bottlenck_first_internal_layer_idx_of_stage(num_blocks)
    for stage_idx in range(4):
        for i in range(num_blocks[stage_idx]):
            internals.append(start_layer_idx_of_stage[stage_idx] + i * 3)
            internals.append(start_layer_idx_of_stage[stage_idx] + 1 + i * 3)
    return internals

def resnet_bottleneck_33_kernel_indices(res_n):
    internals = []
    num_blocks = resnet_n_to_num_blocks[res_n]
    start_layer_idx_of_stage = _resnet_bottlenck_first_internal_layer_idx_of_stage(num_blocks)
    for stage_idx in range(4):
        for i in range(num_blocks[stage_idx]):
            internals.append(start_layer_idx_of_stage[stage_idx] + 1 + i * 3)
    return internals

def resnet_bottleneck_pacesetter_indices(res_n):
    num_blocks = resnet_n_to_num_blocks[res_n]
    start_layer_idx_of_stage = _resnet_bottlenck_first_internal_layer_idx_of_stage(num_blocks)
    return [i-1 for i in start_layer_idx_of_stage]

def resnet_bottleneck_flattened_deps_shrink_by_stage(res_n, shrink_ratio, only_internals=True):
    result_deps = resnet_bottleneck_origin_deps_flattened(res_n=res_n)
    bottleneck_indices = resnet_bottleneck_pacesetter_indices(res_n)
    internals = resnet_bottleneck_internal_kernel_indices(res_n)
    for i in range(len(result_deps)):
        if only_internals and i not in internals:
            continue
        if i >= bottleneck_indices[3]:
            stage_order = 3
        elif i >= bottleneck_indices[2]:
            stage_order = 2
        elif i >= bottleneck_indices[1]:
            stage_order = 1
        elif i >= bottleneck_indices[0]:
            stage_order = 0
        else:
            stage_order = -1
        if stage_order >= 0:
            result_deps[i] = np.ceil(shrink_ratio[stage_order] * result_deps[i])
    result_deps =np.asarray(result_deps, dtype=np.int32)
    print('resnet {} deps shrinked by stage_ratio {} is {}'.format(res_n, shrink_ratio, result_deps))
    return result_deps



def resnet_bottleneck_follow_dict(res_n):
    num_blocks = resnet_n_to_num_blocks[res_n]
    pacesetters = resnet_bottleneck_pacesetter_indices(res_n)
    follow_dict = {}
    for stage_idx in range(4):
        for i in range(num_blocks[stage_idx] + 1):
            follow_dict[pacesetters[stage_idx] + 3 * i] = pacesetters[stage_idx]
    return follow_dict

def resnet_bottleneck_succeeding_strategy(res_n):
    internals = resnet_bottleneck_internal_kernel_indices(res_n)
    pacesetters = resnet_bottleneck_pacesetter_indices(res_n)
    follow_dict = resnet_bottleneck_follow_dict(res_n)
    result = {i : (i+1) for i in internals}
    result[0] = [1,2]
    layers_before_pacesetters = [i - 1 for i in pacesetters]
    for i in follow_dict.keys():
        if i not in pacesetters:
            if i in layers_before_pacesetters:
                result[i] = [i + 1, i + 2]
            else:
                result[i] = i + 1
    return result


####################    WRN
WRN16_FOLLOW_DICT = {1:1, 3:1, 5:1, 6:6, 8:6, 10:6, 11:11, 13:11, 15:11}
WRN16_PACESETTER_IDS = [1, 6, 11]
WRN16_succeeding_STRATEGY = {
    0:[1, 2],
    1:[4, 6, 7],
    2:3,
    4:5,
    6:[9, 11, 12],
    7:8,
    9:10,
    11:[14, 16],
    12:13,
    14:15
}
WRN16_INTERNAL_IDS = [2,4,7,9,12,14]

