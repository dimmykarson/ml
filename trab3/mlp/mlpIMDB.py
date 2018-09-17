#!/usr/bin/python

import numpy as np
import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix


X_train, y_train = load_svmlight_file('saida.txt')
X_test, y_test = load_svmlight_file('IMDBtest.txt')

## save for the confusion matrix
label = y_test

## converts the labels to a categorical one-hot-vector
y_train = keras.utils.to_categorical(y_train, num_classes=2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)

model = Sequential()
# Dense(50) is a fully-connected layer with 50 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 100-dimensional vectors.
model.add(Dense(50, activation='relu', input_dim=100))
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy',
							optimizer='rmsprop',
							metrics=['accuracy'])

model.fit(X_train, y_train, validation_split=0.33, epochs=20, batch_size=128)

score = model.evaluate(X_test, y_test, batch_size=128)
print (score)

y_pred = model.predict_classes(X_test)

cm = confusion_matrix(label, y_pred)
print (cm)

