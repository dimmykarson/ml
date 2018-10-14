#!/usr/bin/python

import numpy as np
import sys
import matplotlib.pyplot as plt
import itertools
import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import History
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(
	cm,
	plot_name,
	target_names,
	title='Confusion matrix',
	cmap=None,
	normalize=True):
	plt.close("all")

	accuracy = np.trace(cm) / float(np.sum(cm))
	misclass = 1 - accuracy

	if cmap is None:
		cmap = plt.get_cmap('Blues')

	plt.figure(figsize=(8, 6))
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()

	if target_names is not None:
		tick_marks = np.arange(len(target_names))
		plt.xticks(tick_marks, target_names, rotation=45)
		plt.yticks(tick_marks, target_names)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


	thresh = cm.max() / 1.5 if normalize else cm.max() / 2
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		if normalize:
			plt.text(j, i, "{:0.4f}".format(cm[i, j]),
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black")
		else:
			plt.text(j, i, "{:,}".format(cm[i, j]),
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black")


	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
	plt.savefig(plot_name+"_cm")


def run(qt_neuronios = 50, qt_camadas = 1, n_epocas = 20):
	plot_name = "results/mlp_Neuronios_{0}_qt_camadas_{1}_epocas_{2}".format(qt_neuronios, qt_camadas, n_epocas)
	
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
		n_neuronios=int(n_neuronios)
	model.add(Dense(2, activation='softmax'))

	model.compile(loss='binary_crossentropy',
								optimizer='rmsprop',
								metrics=['accuracy'])

	model.fit(X_train, y_train, validation_split=0.33, epochs=n_epocas, batch_size=128, callbacks=[history])

	score = model.evaluate(X_test, y_test, batch_size=128)
	print (score)
	with open("results/score_Neuronios_{0}_qt_camadas_{1}_epocas_{2}.txt".format(qt_neuronios, qt_camadas, n_epocas), "+w") as file:
		file.write(str(score))
	y_pred = model.predict_classes(X_test)
	cm = confusion_matrix(label, y_pred)
	print (cm)
	plot_confusion_matrix(cm = cm, plot_name = plot_name, 
		normalize    = True,
		target_names = ['positive', 'negative'],
		title        = "Confusion Matrix")
	print("Ploting")
	plt.close("all")
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig(plot_name)

if __name__ == "__main__":
        print(sys.argv)
        run(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))