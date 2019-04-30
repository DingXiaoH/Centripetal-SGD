import numpy as np


DC40_ORIGIN_DEPS = [16] + [12]*12 + [160] + [12]*12 + [304] + [12]*12   #transition: 13 and 26
DC40_FC_LAYERS = [39]
DC40_SUBSEQUENT_STRATEGY = None
DC40_FOLLOW_DICT = None
DC40_ALL_CONV_LAYERS = range(0, len(DC40_ORIGIN_DEPS))
# completely rewrite the pruning function for DC40!
def customized_dc40_deps(arg):
    if '-rep' in arg:
        arg = arg[:arg.find('-rep')]
    if arg.endswith('-trans'):
        print('customized dc40 deps with transition layers modified')
        return trans_dc40_deps(arg)
    else:
        print('no modification to transition layers')
        deps = np.array(DC40_ORIGIN_DEPS)
        settings = arg.split('-')
        deps[1:13] = int(settings[0])
        deps[14:26] = int(settings[1])
        deps[27:39] = int(settings[2])
        return deps

def trans_dc40_deps(arg):
    assert arg.endswith('-trans')
    deps = np.array(DC40_ORIGIN_DEPS)
    settings = arg.split('-')
    deps[1:13] = int(settings[0])
    deps[13] = deps[0] + int(settings[0]) * 12
    deps[14:26] = int(settings[1])
    deps[26] = deps[13] + int(settings[1]) * 12
    deps[27:39] = int(settings[2])
    return deps