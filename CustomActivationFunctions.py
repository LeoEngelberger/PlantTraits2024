import math

from keras import backend as BK
import keras as ks


def mapping_to_target_range0to1(x, ):
    x02 = ks.ops.relu(x)
    return x02


def mapping_to_target_range0to10(x, ):
    scale = 10.
    x02 = ks.ops.relu(x)
    return x02 * scale


def mapping_to_target_range0to100(x, target_max=100):
    scale = 100.
    x02 = ks.ops.relu(x)
    return x02 * scale


def mapping_to_target_range0to1000(x, target_max=1000):
    scale = 1000.
    x02 = ks.ops.relu(x)
    return x02 * scale


def mapping_to_target_range0to10000(x, ):
    scale = 10000.
    x02 = ks.ops.relu(x)
    return x02 * scale

def map_to_positive(x, ):
    return math.sqrt(x ** 2)