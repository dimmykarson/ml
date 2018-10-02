from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix

def run(params):

	X_train, y_train = load_svmlight_file('digTrain20k.txt')
	X_test, y_test = load_svmlight_file('digTest58k.txt')
	seed = 7

	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	clf = DecisionTreeClassifier()
	num_trees = 2
	model = BaggingClassifier(base_estimator=clf, n_estimators=num_trees, random_state=seed)

	results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold)
	print(results.mean())



run()




