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
from wordcloud import WordCloud
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

train_data = pd.read_csv('train_set.csv', sep="\t")
train_data = train_data
test_data = pd.read_csv('test_set.csv', sep="\t")
test_data = test_data
train_data.columns
set(train_data['Category'])
le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data['Category'])
set(y)
vectorizer=TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,use_idf=True)
#performing Tf-idf transformation and latent semantic indexing
lsa=TruncatedSVD(n_components=20,n_iter=100)
svd_transformer = make_pipeline(vectorizer,lsa)
X = svd_transformer.fit_transform(train_data['Title'])
#Z = svd_transformer.fit_transform(train_data['Title'])
#X=X+Z
terms=vectorizer.get_feature_names()
#for i,comp in enumerate(lsa.components_):
#	TermsInComp=zip(terms,comp)
#	sortedTerms=sorted(TermsInComp,key=lambda x:x[1],reverse=True)[:10]
#	print "Concept %d" %i
#	for term in sortedTerms:
#		print term[0]
   #     print " "
m = RandomForestClassifier()
scores=cross_val_score(m,X,y,cv=10,scoring='accuracy')
score= scores.mean()
predicted = cross_val_predict(m, X, y, cv=10) #10-fold cross validation
print accuracy_score(y, predicted) 
print f1_score(y,predicted,average='micro')
print precision_score(y, predicted, average='macro')
print recall_score(y, predicted, average='weighted')
print classification_report(y, predicted, target_names=list(le.classes_))


