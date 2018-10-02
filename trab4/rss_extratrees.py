from sklearn.ensemble import ExtraTreesClassifier

import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix

def run(params=None):

	X_train, y_train = load_svmlight_file('digTrain20k.txt')
	X_test, y_test = load_svmlight_file('digTest58k.txt')
	seed = 7

	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	model = ExtraTreesClassifier(n_estimators=10)

	results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
	print(results.mean())

run()




