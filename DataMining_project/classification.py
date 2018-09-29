import pandas as pd
from time import time
import random as rand
from sklearn.preprocessing import LabelEncoder
import numpy as np
from numpy import asarray	
import array
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import logging

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize

from sklearn import metrics
from collections import Counter
from time import time
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict

from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

import csv
from plotter import plotter

source="train_set.csv"
df=pd.read_csv(source,sep="\t")

t0=time()
vectorizer=CountVectorizer(stop_words='english',min_df=0)
transformer=TfidfTransformer(sublinear_tf=True)
cmatrix=vectorizer.fit_transform(df['Content'])
tfmatrix=transformer.fit_transform(cmatrix)
svd=TruncatedSVD(n_components=30)
X_train=svd.fit_transform(tfmatrix)

X_train_countvect=cmatrix
print "Vectorising Lasted:",time()-t0

#binarized Y_train

lb=LabelBinarizer()
Y_train_binarized = lb.fit_transform(df['Category'])    

name=['Metrics','SVM','RandomForest','NearestNeighbors','MultinomialNB']
accuracy=['Accuracy']
auc=['AUC']
recall=['Recall']
precision=['Precision']
f_measure=['F-Measure']

#init plotter for roc
pl=plotter()
pl.set_classes("train_set.csv")

#SVC Classifier test
t0=time()
clf=LinearSVC(C=4.0, random_state=42)
multicls=OneVsRestClassifier(clf)
SGDpredicted=cross_val_predict(estimator=multicls,X=X_train,y=Y_train_binarized,cv=10)
print "SVC classifier Test Lasted:",time()-t0
accuracy.append(metrics.accuracy_score(Y_train_binarized,SGDpredicted))
returnval=metrics.precision_recall_fscore_support(Y_train_binarized, SGDpredicted)
precision.append(np.mean(returnval[0]))
recall.append(np.mean(returnval[1]))
f_measure.append(np.mean(returnval[2]))
auc.append(pl.roc(Y_train_binarized,SGDpredicted,name[1]))

#RandomForest test
t0=time()
clf=RandomForestClassifier(n_estimators=5)
multicls=OneVsRestClassifier(clf)

RFpredicted=cross_val_predict(estimator=multicls,X=X_train,y=Y_train_binarized,cv=10)
#print 'RandomForest Test Accuracy' ,accuracy[1]
print "RandomForest classifier Test Lasted:",time()-t0
accuracy.append(metrics.accuracy_score(Y_train_binarized,RFpredicted))
returnval=metrics.precision_recall_fscore_support(Y_train_binarized, RFpredicted)
precision.append(np.mean(returnval[0]))
recall.append(np.mean(returnval[1]))
f_measure.append(np.mean(returnval[2]))
auc.append(pl.roc(Y_train_binarized,RFpredicted,name[2]))

#KNN test
t0=time()
clf=KNeighborsClassifier()
multicls=OneVsRestClassifier(clf)

KNpredicted=cross_val_predict(estimator=multicls,X=X_train,y=Y_train_binarized,cv=10)
#auc.append(metrics.auc(X_train,Y_train_binarized))
#print 'KNN Test Accuracy',accuracy[2]
print "KNN classifier Test Lasted:",time()-t0
accuracy.append(metrics.accuracy_score(Y_train_binarized,KNpredicted))
returnval=metrics.precision_recall_fscore_support(Y_train_binarized, KNpredicted)
precision.append(np.mean(returnval[0]))
recall.append(np.mean(returnval[1]))
f_measure.append(np.mean(returnval[2]))
auc.append(pl.roc(Y_train_binarized,KNpredicted,name[3]))

#Naive Bayes (Multimomial)
t0=time()
clf=MultinomialNB()
multicls=OneVsRestClassifier(clf)

MNBpredicted=cross_val_predict(estimator=multicls,X=X_train_countvect,y=Y_train_binarized,cv=10)
#print 'MultinomialNB Test Accuracy', accuracy[3]
print "Naive Bayes classifier Test Lasted:",time()-t0
accuracy.append(metrics.accuracy_score(Y_train_binarized,MNBpredicted))
returnval=metrics.precision_recall_fscore_support(Y_train_binarized, MNBpredicted)
precision.append(np.mean(returnval[0]))
recall.append(np.mean(returnval[1]))
f_measure.append(np.mean(returnval[2]))
auc.append(pl.roc(Y_train_binarized,MNBpredicted,name[4]))


print(accuracy)
print(precision)
print(recall)
print(f_measure)

#write results in metrics.csv
myfile = open('EvaluationMetric_10fold.csv', 'wb')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
wr.writerow(name)
wr.writerow(accuracy)
wr.writerow(precision)
wr.writerow(recall)
wr.writerow(f_measure)

pl.Show()
