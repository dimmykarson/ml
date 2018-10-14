import sys
import numpy as np
import pandas
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from util import plot_residual

name = "Linear"

def run(target='f3', normalizar='False'):
	data = pandas.read_csv("usina72.csv")
	feature_cols = ['f5', 'f6', 'f7', 'f9', 'f10', 'f11', 'f12']
	X = data.loc[:,feature_cols]
	print("Avaliando {0}".format(target))
	label_col = [target]
	Y = data.loc[:,label_col]
	X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.5, random_state=0)
	if normalizar=='True':
		print("Normalizando...")
		scaler = preprocessing.MinMaxScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.fit_transform(X_test)

	reg = LinearRegression().fit(X_train, y_train)	
	y_pred = reg.predict(X_test)
	plot_name = "results/{0} {1} {2}".format(name, target, normalizar)
	MSE = mean_squared_error(y_test.values.ravel(), y_pred)
	print(MSE)
	file = open(plot_name+".txt", "w")
	file.write("MSE {0}\n".format(MSE))
	file.close()
	plot_residual(res=y_test.values.ravel()-y_pred, plot_name=plot_name, title="Residuos "+name)

if __name__ == "__main__":
	print(sys.argv)
	run(sys.argv[1], sys.argv[2])


