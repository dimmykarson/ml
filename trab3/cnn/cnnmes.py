#!/usr/bin/python


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from PIL import Image
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
import sys
import matplotlib.pyplot as plt
import itertools
import pandas
from keras.callbacks import History
batch_size = 128
num_classes = 12
train_file = './mes/train.txt'
test_file = './mes/test.txt'
qt_cnn=5
funcao = 'tanh'
kernel = (4, 4)
#MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, 
tipo_pooling = 'MaxPooling2D'
n_epocas=25
n_neuronios_mlp=128
n_camadas_mlp=4

# input image dimensions
img_rows, img_cols = 64, 64

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


#==========================================================================

def load_images(image_paths, convert=False):

	x = []
	y = []
	for image_path in image_paths:

		path, label = image_path.split(' ')
		
		path= './mes/data/' + path 

		if convert:
			image_pil = Image.open(path).convert('RGB') 
		else:
			image_pil = Image.open(path).convert('L')

		img = np.array(image_pil, dtype=np.uint8)

		x.append(img)
		y.append([int(label)])


	x = np.array(x)
	y = np.array(y)

	if np.min(y) != 0: 
		y = y-1

	return x, y
	
	

def load_dataset(train_file, test_file, resize, convert=False, size=(224,224)):

	arq = open(train_file, 'r')
	texto = arq.read()
	train_paths = texto.split('\n')
	
	print('Size : ', size)

	train_paths.remove('') #remove empty lines
	train_paths.sort()
	x_train, y_train = load_images(train_paths, convert)

	arq = open(test_file, 'r')
	texto = arq.read()
	test_paths = texto.split('\n')

	test_paths.remove('') #remove empty lines
	test_paths.sort()
	x_test, y_test = load_images(test_paths, convert)

	if resize:
		print("Resizing images...")
		x_train = resize_data(x_train, size, convert)
		x_test = resize_data(x_test, size, convert)

	if not convert:
		x_train = x_train.reshape(x_train.shape[0], size[0], size[1], 1)
		x_test = x_test.reshape(x_test.shape[0], size[0], size[1], 1)


	print(np.shape(x_train))
	return (x_train, y_train), (x_test, y_test)

def resize_data(data, size, convert):

	if convert:
		data_upscaled = np.zeros((data.shape[0], size[0], size[1], 3))
	else:
		data_upscaled = np.zeros((data.shape[0], size[0], size[1]))
	for i, img in enumerate(data):
		large_img = cv2.resize(img, dsize=(size[1], size[0]), interpolation=cv2.INTER_CUBIC)
		data_upscaled[i] = large_img

	print(np.shape(data_upscaled))
	return data_upscaled

#==========================================================================

def run():
	print("Loading database...")
	# gray scale
	#input_shape = (img_rows, img_cols, 1)
	#(x_train, y_train), (x_test, y_test) = load_dataset(train_file, test_file, resize=True, convert=False, size=(img_rows, img_cols))

	# rgb
	input_shape = (img_rows, img_cols, 3)
	(x_train, y_train), (x_test, y_test) = load_dataset(train_file, test_file, resize=True, convert=True, size=(img_rows, img_cols))

	### save for the confusion matrix
	label = []
	for i in range(len(x_test)):
		label.append(y_test[i][0])
	plot_name = "results/mes/cnn_{0}_epocas_{1} MLP {2} {3} Fun {4} Filt {5} Poo {6} ".format(qt_cnn, n_epocas, n_neuronios_mlp, n_camadas_mlp, funcao, kernel, tipo_pooling)
	#normalize images
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape)

	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	# create cnn model
	model = Sequential()
	history = History()
	model.add(Conv2D(32, kernel_size=(3, 3), activation=funcao, input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation=funcao))
	pooling = MaxPooling2D(pool_size=(2, 2))
	if tipo_pooling == 'AveragePooling2D':
		pooling = AveragePooling2D(pool_size=(2, 2))
	elif tipo_pooling == 'GlobalMaxPooling2D':
		pooling = GlobalMaxPooling2D()
	model.add(pooling)
	model.add(Conv2D(128, kernel, activation=funcao))
	model.add(pooling)
	model.add(Conv2D(128, kernel, activation=funcao))
	model.add(pooling)
	model.add(Conv2D(128, kernel, activation=funcao))
	model.add(pooling)
	model.add(Dropout(0.25))
	model.add(Flatten())
	n_neuronios = n_neuronios_mlp
	for i in range(n_camadas_mlp):
		model.add(Dense(n_neuronios, activation=funcao))
		n_neuronios = int(n_neuronios/2)
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	# print cnn layers
	print('Network structure ----------------------------------')
	for i, layer in enumerate(model.layers):
		print(i,layer.name)
		if hasattr(layer, 'output_shape'):
			print(layer.output_shape)
	print('----------------------------------------------------')

	model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

	model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epocas, verbose=1, validation_data=(x_test, y_test), callbacks=[history])

	score = model.evaluate(x_test, y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	with open("results/mes/score_cnn_{0}_epocas_{1} MLP {2} {3} Fun {4} Filt {5} Poo {6} ".format(qt_cnn, n_epocas, n_neuronios_mlp, n_camadas_mlp, funcao, kernel, tipo_pooling), "+w") as file:
			file.write(str(score))
	#print model.predict_classes(x_test) #classes predicted
	#print model.predict_proba(x_test) #classes probability

	pred = []
	y_pred = model.predict_classes(x_test)
	for i in range(len(x_test)):
		pred.append(y_pred[i])

	cm = confusion_matrix(label, pred)
	print(cm)
	plot_confusion_matrix(cm = cm, plot_name = plot_name, 
		normalize    = True,
		target_names = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez" ],
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
        run()