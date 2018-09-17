#!/usr/bin/python

import numpy as np
import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import History
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

qt_neuronios = 50
qt_camadas = 1
n_epocas = 20


X_train, y_train = load_svmlight_file('IMDBtrain.txt')
X_test, y_test = load_svmlight_file('IMDBtest.txt')

## save for the confusion matrix
label = y_test

## converts the labels to a categorical one-hot-vector
y_train = keras.utils.to_categorical(y_train, num_classes=2)
y_test = keras.utils.to_categorical(y_test, num_classes=2)

model = Sequential()
history = History()
# Dense(50) is a fully-connected layer with 50 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 100-dimensional vectors.
n_neuronios = qt_neuronios

for i in range(0, qt_camadas):
	model.add(Dense(n_neuronios, activation='relu', input_dim=100))
	n_neuronios=n_neuronios/2
model.add(Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy',
							optimizer='rmsprop',
							metrics=['accuracy'])

model.fit(X_train, y_train, validation_split=0.33, epochs=n_epocas, batch_size=128, callbacks=[history])

score = model.evaluate(X_test, y_test, batch_size=128)
print (score)
y_pred = model.predict_classes(X_test)
cm = confusion_matrix(label, y_pred)
print (cm)

print("Ploting")
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plot_name = "mlp_Neuronios_{0}_qt_camadas_{1}_epocas_{2}".format(qt_neuronios, qt_camadas, n_epocas)
plt.savefig(plot_name)
plt.show()

