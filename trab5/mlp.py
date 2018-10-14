import sys
import numpy as np
import pandas
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense

from util import plot_residual

name="MLP"

def run(target='f3', normalizar='False'):
	data = pandas.read_csv("usina72.csv")
	feature_cols = ['f5', 'f6', 'f7', 'f9', 'f10', 'f11', 'f12']
	X = data.loc[:,feature_cols]
	print("Avaliando {0}".format(target))
	label_col = [target]
	Y = data.loc[:,label_col]
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
	if normalizar == 'True':
		print("Normalizando...")
		scaler = preprocessing.MinMaxScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.fit_transform(X_test)
	
	model = Sequential()
	model.add(Dense(9, input_dim=8, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(X_train, y_train, epochs=100, validation_split=0.33, verbose=1)
	y_pred = model.predict(X_test)

	MSE = mean_squared_error(y_test.values.ravel(), y_pred)
	print("MSE: {0}".format(MSE))
	res = y_test - y_pred
	plot_name = "residuos_"+name+"_target_{0}_norm_{1}".format(target, normalizar)
	file = open("results/"+plot_name+".txt", "w")
	file.write("MSE {0}\n".format(MSE))
	file.close()
	plot_residual(res=res, plot_name=plot_name, title='Residuos '+name, show=False)

if __name__ == "__main__":
	print(sys.argv)
	run(sys.argv[1], sys.argv[2])



