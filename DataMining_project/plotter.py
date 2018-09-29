import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn import metrics


colors=['aqua', 'darkorange', 'cornflowerblue','navy','deeppink','black']

class plotter(object):

	def __init__(self):
		super(plotter, self).__init__()
		self.position=0
		self.classes=0
		plt.figure()

	def set_classes(self, source):
		df=pd.read_csv(source,sep="\t")
		A=np.array(df)
		self.classes=A.shape[1] 

	def roc(self, y_train, y_predicted, name):
		self.position = self.position + 1
		fpr = dict()
		tpr = dict()
		roc_auc = dict()
		for i in range(self.classes):
			fpr[i], tpr[i], _ = roc_curve(y_train.ravel(), y_predicted.ravel())
			roc_auc[i] = auc(fpr[i], tpr[i])

		fpr['micro'], tpr['micro'], _  = roc_curve(y_train.ravel(), y_predicted.ravel())
		roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

		lw=2
		plt.plot(fpr['micro'], tpr['micro'], label=name+' micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc['micro']), color = colors[self.position%6])
		plt.plot([0, 1], [0, 1], 'k--', lw=lw)
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel("False Positive Rate")
		plt.ylabel('True Positive Rate')
		plt.title('Some extension of ROC')
		plt.legend(loc="lower right")
		return roc_auc["micro"]

	def Show(self):
		plt.show()