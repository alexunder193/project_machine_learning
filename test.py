from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn import svm
from wordcloud import WordCloud
import pandas as pd
import os
import matplotlib.pyplot as plt

train_data = pd.read_csv('train_set.csv', sep="\t")
train_data = train_data[25:50]
test_data = pd.read_csv('test_set.csv', sep="\t")
test_data = test_data[30:55]
train_data.columns
set(train_data['Category'])
train_data
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])
set(y)
#print ("categories:")
#print (y)
#set(le.inverse_transform(y))
#print (set(le.inverse_transform(y)))

count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
print (count_vectorizer)
X = count_vectorizer.fit_transform(test_data['Content'])
print ("contents:")
#print (X)
print (X.shape)
print (X.toarray())
#print (X)
clf=svm.SVC(C=1000, gamma=1, kernel='linear')
clf = RandomForestClassifier()
clf.fit(X,y)
y_pred = clf.predict(X)
print (y_pred)
predicted_categories = le.inverse_transform(y_pred)
print (predicted_categories)
print classification_report(y, y_pred, target_names=list(le.classes_))

train_data = pd.read_csv('train_set.csv', sep="\t")
train_data = train_data[0:25]
test_data = pd.read_csv('test_set.csv', sep="\t")
test_data = test_data[25:50]
train_data.columns
set(train_data['Category'])
train_data
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])
set(y)
count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
X_train = count_vectorizer.fit_transform(train_data['Content'])
X_test = count_vectorizer.fit_transform(test_data['Content'])
print (X_train.shape)
print (X_train.toarray())
clf=svm.SVC(C=1000, gamma=1, kernel='linear')
clf.fit(X_train,y)
y_pred = clf.predict(X_test)
print (y_pred)
predicted_categories = le.inverse_transform(y_pred)
print (predicted_categories)
print classification_report(y, y_pred, target_names=list(le.classes_))



