import numpy as np
import random
import config

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