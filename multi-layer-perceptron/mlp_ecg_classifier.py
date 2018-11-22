import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from utils import load_data, as_keras_metric, precision, recall, fmeasure
import keras_metrics
from keras.utils import plot_model
from keras import regularizers
from sklearn.metrics import classification_report
from confusion_matrix import plot_confusion_matrix_from_data

class_names = ['Normal', 'Supraventricular ectopic beat', 'Ventricular ectopic beat', 'Fusion beat', 'Unknown beat']
num_classes = 5
num_epochs = 100
batch_size = 256
data_root = '../data/'
num_features = (187)

(x_train, y_train), (x_test, y_test) = load_data(data_root)

print("Train shapes:", x_train.shape, y_train.shape)
print("Test shapes:", x_test.shape, y_test.shape)

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()

model.add(Dense(units=256, use_bias=True, bias_initializer='zeros', activation='relu', input_dim=num_features))
model.add(Dense(units=128, use_bias=True, bias_initializer='zeros', activation='relu', input_dim=num_features))
model.add(Dense(units=64, use_bias=True, bias_initializer='zeros', activation='relu'))
model.add(Dense(units=32, use_bias=True, bias_initializer='zeros', activation='relu'))
model.add(Dense(units=num_classes, use_bias=True, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy', precision, recall])

model.fit(x=x_train, y=y_train, epochs=num_epochs, batch_size=batch_size, shuffle=True)
# plot_model(model, to_file='vis.png')

y_pred = model.predict_classes(x_test)
print(y_pred)
y_test = np.argmax(y_test, axis=1) # Convert one-hot to index
plot_confusion_matrix_from_data(y_test, y_pred, columns=class_names)
print(classification_report(y_test, y_pred))

# score = model.evaluate(x_test, y_test)
# print("Test loss: ", score[0])
# print("Test accuracy: ", score[1])
