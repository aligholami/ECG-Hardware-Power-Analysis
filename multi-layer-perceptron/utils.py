from keras.utils import to_categorical
import pandas as pd
import config

def load_data(data_path):

    x_train, y_train = [], []
    x_test, y_test = [], []

    train_df = pd.read_csv(data_path + config.TRAIN_FILE_NAME)
    test_df = pd.read_csv(data_path + config.TEST_FILE_NAME)

    y_train = train_df.iloc[:, -1]
    y_test = test_df.iloc[:, -1]

    return (x_train, y_train), (x_test, y_test)