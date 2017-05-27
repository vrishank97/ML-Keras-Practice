from keras.models import Sequential
from keras.layers import Dense
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd
import keras

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
x_train = mnist.train.images
y_train = mnist.train.labels

model = Sequential()
model.add(Dense(20, input_dim=784, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train)

x_test = mnist.test.images
y_test = mnist.test.labels

scores = model.evaluate(x_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save('my_model.h5')