import pandas as pd
import config

def load_data(data_path):

    x_train, y_train = [], []
    x_test, y_test = [], []
    
    train_df = pd.read_csv(data_path + config.TRAIN_FILE_NAME)
    test_df = pd.read_csv(data_path + config.TEST_FILE_NAME)

    # Extract the labels
    y_train = train_df.iloc[:, -1]
    y_test = test_df.iloc[:, -1]

    # Extract data
    x_train = train_df.iloc[:, :-1] 
    x_test = test_df.iloc[:, :-1]

    return (x_train, y_train), (x_test, y_test)

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