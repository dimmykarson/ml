import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm,
	target_names,
	title='Confusion matrix',
	cmap=None,
	normalize=True):


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
	plt.show()


array = [[2754,1,2,3,1,2,14,1,2,0]
 ,[0,3308,2,3,1,2,3,8,0,3]
 ,[1,6,2908,4,4,0,0,12,6,3]
 ,[3,2,4,2864,0,13,1,13,7,2]
 ,[0,6,1,0,2816,0,4,6,1,27]
 ,[1,2,0,9,1,2736,12,0,4,4]
 ,[5,7,3,0,7,8,2899,0,0,0]
 ,[0,6,8,2,3,0,0,3023,0,6]
 ,[5,7,9,3,3,5,8,7,2789,11]
 ,[3,2,2,6,27,3,1,11,8,2843]]

plot_confusion_matrix(cm           = np.array(array), 
                      normalize    = False,
                      target_names = ['high', 'medium', 'low'],
                      title        = "Confusion Matrix")