from sklearn.ensemble import AdaBoostClassifier
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix
from util import plot_confusion_matrix
import timeit

def run(num=2, lr=1):
	start = timeit.default_timer()
	X_train, y_train = load_svmlight_file('digTrain20k.txt')
	X_test, y_test = load_svmlight_file('digTest58k.txt')
	seed = 7

	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	model = AdaBoostClassifier(n_estimators=num, learning_rate=lr,random_state=None)

	results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
	model.fit(X_train, y_train)
	predict = model.predict(X_test)
	score = model.score(X_test, y_test)
	cm = confusion_matrix(y_test, predict)
	stop = timeit.default_timer()
	print("Estimators %d %f.4 Time: %s " % (num, score, str(stop - start)))
	plot_confusion_matrix(cm = cm, plot_name = "results/bagging/ada/bagging_ada_{0}_{1}".format(num, lr), 
		normalize    = True,
		target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
		title        = "Confusion Matrix")
	
if __name__ == "__main__":
	print(sys.argv)
	run(int(sys.argv[1]), int(sys.argv[2]))


