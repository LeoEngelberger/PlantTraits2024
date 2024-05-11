from keras import backend as BK
import keras as ks

def mapping_to_target_range0to1( x, target_min=0, target_max=1 ) :
    x02 = ks.ops.tanh(x) + 1 # x in range(0,2)
    scale = ( target_max-target_min )/2.
    return  x02 * scale + target_min
def mapping_to_target_range0to10( x, target_min=0, target_max=10 ) :
    x02 = ks.ops.tanh(x) + 1 # x in range(0,2)
    scale = ( target_max-target_min )/2.
    return  x02 * scale + target_min
def mapping_to_target_range0to100( x, target_min=0, target_max=100 ) :
    x02 = ks.ops.tanh(x) + 1 # x in range(0,2)
    scale = ( target_max-target_min )/2.
    return  x02 * scale + target_min
def mapping_to_target_range0to1000( x, target_min=0, target_max=1000 ) :
    x02 = ks.ops.tanh(x) + 1 # x in range(0,2)
    scale = ( target_max-target_min )/2.
    return  x02 * scale + target_min
def mapping_to_target_range0to10000( x, target_min=0, target_max=10000 ) :
    x02 = ks.ops.tanh(x) + 1 # x in range(0,2)
    scale = ( target_max-target_min )/2.
    return  x02 * scale + target_min