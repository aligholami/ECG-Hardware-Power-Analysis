import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from utils import load_data

class_names = ['N', 'S', 'V', 'F', 'Q']
num_classes = 5
num_epochs = 20
batch_size = 128
data_root = './data/'
input_shape = (188, 1)

(x_train, y_train), (x_test, y_test) = load_data(data_root)

model = Sequential()

model.add(Dense(units=128, activation='relu', input_shape=input_shape))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs=num_epochs, batch_size=batch_size)

score = model.evaluate(x_test, y_test)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])
