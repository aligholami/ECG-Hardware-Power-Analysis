import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from utils import load_data

class_names = ['N', 'S', 'V', 'F', 'Q']
num_classes = 6
num_epochs = 20
batch_size = 128
data_root = './data/'

(x_train, y_train), (x_test, y_test) = load_data(data_root)

model = Sequential()

model.add(Dense(units=128, activation='relu', input_dim=_))
model.add(Dense(units=64, activation='relu', input_dim=_))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs=num_epochs, batch_size=batch_size)

loss_metrics = model.evaluate(x=x_test, y=y_test, batch_size=batch_size)
