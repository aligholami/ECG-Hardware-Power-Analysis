from keras import backend as K
import pandas as pd
import numpy as np
import config
import augmenter
from sklearn.utils import shuffle

def load_data(data_path):
    
    train_df = pd.read_csv(data_path + config.TRAIN_FILE_NAME, header=None)
    test_df = pd.read_csv(data_path + config.TEST_FILE_NAME, header=None)
    df = pd.concat([train_df, test_df], axis=0)

    print(df.head())
    # Augment test data by n
    M = df.values
    X = M[:, :-1]
    y = M[:, -1].astype(int)

    C0 = np.argwhere(y == 0).flatten()
    C1 = np.argwhere(y == 1).flatten()
    C2 = np.argwhere(y == 2).flatten()
    C3 = np.argwhere(y == 3).flatten()
    C4 = np.argwhere(y == 4).flatten()
    print(C0)
    print(C1)

    augment = augmenter.augment_by_n

    result = np.apply_along_axis(augment, axis=1, arr=X[C3]).reshape(-1, 187)
    classe = np.ones(shape=(result.shape[0],), dtype=int)*3
    X = np.vstack([X, result])
    y = np.hstack([y, classe])

    subC0 = np.random.choice(C0, 1200)
    subC1 = np.random.choice(C1, 1200)
    subC2 = np.random.choice(C2, 1200)
    subC3 = np.random.choice(C3, 1200)
    subC4 = np.random.choice(C4, 1200)

    X_test = np.vstack([X[subC0], X[subC1], X[subC2], X[subC3], X[subC4]])
    y_test = np.hstack([y[subC0], y[subC1], y[subC2], y[subC3], y[subC4]])

    X_train = np.delete(X, [subC0, subC1, subC2, subC3, subC4], axis=0)
    y_train = np.delete(y, [subC0, subC1, subC2, subC3, subC4], axis=0)

    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    X_test, y_test = shuffle(X_test, y_test, random_state=0)

    return (X_train, y_train), (X_test, y_test)

def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.

    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
        
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)