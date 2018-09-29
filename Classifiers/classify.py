import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS,CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn import preprocessing,cross_validation,grid_search
from sklearn.model_selection import KFold,GridSearchCV ,cross_val_predict# import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.multiclass import OneVsRestClassifier
from wordcloud import WordCloud
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

train_data = pd.read_csv('train_set.csv', sep="\t")
train_data = train_data[0:10000]
test_data = pd.read_csv('test_set.csv', sep="\t")
test_data = test_data[0:3000]
train_data.columns
set(train_data['Category'])
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data['Category'])
set(y)

name=['Statistic Measure','Naive Bayes','Random Forest','SVM','KNN','My Method']
accuracy=['Accuracy']
precision=['Precision']
recall=['Recall']
f_measure=['F-Measure']

vectorizer=TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,use_idf=True)
#performing Tf-idf transformation and latent semantic indexing
lsa=TruncatedSVD(n_components=20)
svd_transformer = make_pipeline(vectorizer,lsa)
X = svd_transformer.fit_transform(train_data['Content']+10*train_data['Title'])
#splitting the train set and the test set
X_train, X_test, y_train, y_test = train_test_split(             
   X, y, test_size=0.5, random_state=0)

#SVM
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
scores = ['precision', 'recall']
clf = GridSearchCV(SVC(), tuned_parameters, cv=10)
clf.fit(X_train, y_train)
y_true, y_pred = y_test, clf.predict(X_test)
#for score in scores:
    #print("# Tuning hyper-parameters for %s" % score)
    #print()
    #clf = GridSearchCV(SVC(), tuned_parameters, cv=10,scoring='%s_macro' % score)
    #clf.fit(X_train, y_train)
    #print("Best parameters set found on development set:")
    #print()
    #print(clf.best_params_)
    #print()
    #print("Grid scores on development set:")
    #print()
    #y_true, y_pred = y_test, clf.predict(X_test)
    #print classification_report(y_true, y_pred, target_names=list(le.classes_))
acc=accuracy_score(y_true, y_pred)
accuracy.append(acc)
pre=precision_score(y_true, y_pred,average='macro')
precision.append(pre)
rec=recall_score(y_true, y_pred,average='macro')
recall.append(rec)
f1=f1_score(y_true, y_pred,average='macro')
f_measure.append(f1)
print (accuracy)
print (recall)
print (precision)
print (f_measure)
#Random Forest
m = RandomForestClassifier()
scores=cross_val_score(m,X,y,cv=10,scoring='accuracy')
score= scores.mean()
predicted = cross_val_predict(m, X, y, cv=10) #10-fold cross validation
acc=accuracy_score(y, predicted)
accuracy.append(acc)
pre=precision_score(y, predicted,average='macro')
precision.append(pre)
rec=recall_score(y,predicted,average='macro')
recall.append(rec)
f1=f1_score(y,predicted,average='macro')
f_measure.append(f1)
print (accuracy)
print (recall)
print (precision)
print (f_measure)
#Naive Bayes
X_naive=vectorizer.fit_transform(train_data['Content']+5*train_data['Title'])
clf=MultinomialNB()
multicls=OneVsRestClassifier(clf)
MNBpredicted=cross_val_predict(estimator=multicls,X=X_naive,y=y,cv=10)
acc=accuracy_score(y,MNBpredicted)
accuracy.append(acc)
pre=precision_score(y, MNBpredicted,average='macro')
precision.append(pre)
rec=recall_score(y,MNBpredicted,average='macro')
recall.append(rec)
f1=f1_score(y,MNBpredicted,average='macro')
f_measure.append(f1)
print (accuracy)
print (recall)
print (precision)
print (f_measure)

myfile = open('EvaluationMetric_10fold.csv', 'wb')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
wr.writerow(name)
wr.writerow(accuracy)
wr.writerow(precision)
wr.writerow(recall)
wr.writerow(f_measure)
