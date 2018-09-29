import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS,CountVectorizer,TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing,cross_validation,grid_search
from sklearn.model_selection import KFold,GridSearchCV # import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from wordcloud import WordCloud
import pandas as pd
import os
import matplotlib.pyplot as plt

train_data = pd.read_csv('train_set.csv', sep="\t")
train_data = train_data[0:300]
test_data = pd.read_csv('test_set.csv', sep="\t")
test_data = test_data[0:300]
train_data.columns
set(train_data['Category'])
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data['Category'])
set(y)
#y=np.array(y)
#print y
count_vectorizer = CountVectorizer(stop_words=ENGLISH_STOP_WORDS)
X = count_vectorizer.fit_transform(train_data['Content'])
X.toarray()
#splitting the train set and the test set
X_train, X_test, y_train, y_test = train_test_split(             
   X, y, test_size=0.5, random_state=0)
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=10,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print classification_report(y_true, y_pred, target_names=list(le.classes_))
    





