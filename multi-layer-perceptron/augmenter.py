import numpy as np
from scipy.signal import resample
import random
import config

def stretch(x):
    l = int(187 * (1 + (random.random()-0.5)/3))
    y = resample(x, l)
    if l < 187:
        y_ = np.zeros(shape=(187, ))
        y_[:l] = y
    else:
        y_ = y[:187]
    return y_

def amplify(x):
    alpha = (random.random()-0.5)
    factor = -alpha*x + (1+alpha)
    return x*factor

def augment_by_n(x):
    augmented = np.zeros(shape=(config.AUGMENT_N_TIMES, config.NUM_FEATURES))

    for i in range(config.AUGMENT_N_TIMES - 1):
        if random.random() < 0.33:
            new_y = stretch(x)

        elif random.random() < 0.66:
            new_y = amplify(x)

        else:
            new_y = stretch(x)
            new_y = amplify(x)
    
        augmented[i, :] = new_y
    return augmented