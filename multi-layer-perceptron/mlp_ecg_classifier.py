import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from utils import load_data, as_keras_metric
import keras_metrics
from keras.utils import plot_model
from keras import regularizers

class_names = ['N', 'S', 'V', 'F', 'Q']
num_classes = 5
num_epochs = 20
batch_size = 128
data_root = '../data/'
num_features = (187)

(x_train, y_train), (x_test, y_test) = load_data(data_root)

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()

model.add(Dense(units=128, activation='relu', input_dim=num_features))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

# Load special metrics
precision = as_keras_metric(tf.metrics.precision)
recall = as_keras_metric(tf.metrics.recall)


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.2)
# plot_model(model, to_file='vis.png')

score = model.evaluate(x_test, y_test)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])
