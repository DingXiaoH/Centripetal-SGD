import numpy as np
from constants import *

def get_con_flops(input_deps, output_deps, h, w=None, kernel_size=3, groups=1):
    if w is None:
        w = h
    return input_deps * output_deps * h * w * kernel_size * kernel_size // groups

def calculate_rc_flops(deps, rc_n):
    result = []
    result.append(get_con_flops(3, deps[0], 32, 32))
    for i in range(rc_n):
        result.append(get_con_flops(deps[2*i], deps[2*i+1], 32, 32))
        result.append(get_con_flops(deps[2*i+1], deps[2*i+2], 32, 32))

    project_layer_idx = 2 * rc_n + 1
    result.append(get_con_flops(deps[project_layer_idx - 1], deps[project_layer_idx], 16, 16, 2))
    result.append(get_con_flops(deps[project_layer_idx - 1], deps[project_layer_idx + 1], 16, 16))
    result.append(get_con_flops(deps[project_layer_idx + 1], deps[project_layer_idx + 2], 16, 16))
    for i in range(rc_n - 1):
        result.append(get_con_flops(deps[2 * i + project_layer_idx + 2], deps[2 * i + project_layer_idx + 3], 16, 16))
        result.append(get_con_flops(deps[2 * i + project_layer_idx + 3], deps[2 * i + project_layer_idx + 4], 16, 16))

    project_layer_idx += 2 * rc_n + 1
    result.append(get_con_flops(deps[project_layer_idx - 1], deps[project_layer_idx], 8, 8, 2))
    result.append(get_con_flops(deps[project_layer_idx - 1], deps[project_layer_idx + 1], 8, 8))
    result.append(get_con_flops(deps[project_layer_idx + 1], deps[project_layer_idx + 2], 8, 8))
    for i in range(rc_n - 1):
        result.append(get_con_flops(deps[2 * i + project_layer_idx + 2], deps[2 * i + project_layer_idx + 3], 8, 8))
        result.append(get_con_flops(deps[2 * i + project_layer_idx + 3], deps[2 * i + project_layer_idx + 4], 8, 8))

    result.append(10*deps[-1])
    return np.sum(result)


def calculate_resB_bottleneck_flops(fd, resnet_n):
    num_blocks = resnet_n_to_num_blocks[resnet_n]
    d = convert_resnet_bottleneck_deps(fd)
    result = []
    # conv1
    result.append(get_con_flops(3, d[0], 112, 112, kernel_size=7))
    last_dep = d[0]
    # stage 2
    result.append(get_con_flops(last_dep, d[1][0][2], 56, kernel_size=1))
    for i in range(num_blocks[0]):
        result.append(get_con_flops(last_dep, d[1][i][0], 56, kernel_size=1))
        result.append(get_con_flops(d[1][i][0], d[1][i][1], 56, kernel_size=3))
        result.append(get_con_flops(d[1][i][1], d[1][i][2], 56, kernel_size=1))
        last_dep = d[1][i][2]
    # stage 3
    result.append(get_con_flops(last_dep, d[2][0][2], 28, kernel_size=1))
    for i in range(num_blocks[1]):
        result.append(get_con_flops(last_dep, d[2][i][0], 56 if i == 0 else 28, kernel_size=1))
        result.append(get_con_flops(d[2][i][0], d[2][i][1], 28, kernel_size=3))
        result.append(get_con_flops(d[2][i][1], d[2][i][2], 28, kernel_size=1))
        last_dep = d[2][i][2]
    # stage 4
    result.append(get_con_flops(last_dep, d[3][0][2], 14, kernel_size=1))
    for i in range(num_blocks[2]):
        result.append(get_con_flops(last_dep, d[3][i][0], 28 if i == 0 else 14, kernel_size=1))
        result.append(get_con_flops(d[3][i][0], d[3][i][1], 14, kernel_size=3))
        result.append(get_con_flops(d[3][i][1], d[3][i][2], 14, kernel_size=1))
        last_dep = d[3][i][2]
    # stage 5
    result.append(get_con_flops(last_dep, d[4][0][2], 7, kernel_size=1))
    for i in range(num_blocks[3]):
        result.append(get_con_flops(last_dep, d[4][i][0], 14 if i == 0 else 7, kernel_size=1))
        result.append(get_con_flops(d[4][i][0], d[4][i][1], 7, kernel_size=3))
        result.append(get_con_flops(d[4][i][1], d[4][i][2], 7, kernel_size=1))
        last_dep = d[4][i][2]
    # fc
    result.append(1000 * last_dep)
    return np.sum(np.array(result, dtype=np.float32))

#   fd : flattened deps
def calculate_resB_50_flops(fd):
    return calculate_resB_bottleneck_flops(fd, 50)

def calculate_rc56_flops(deps):
    return calculate_rc_flops(deps, 9)
def calculate_rc110_flops(deps):
    return calculate_rc_flops(deps, 18)