import pandas as pd
from time import time
import random as rand

from sklearn.preprocessing import LabelEncoder
import numpy as np
from numpy import asarray	
import array

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import linear_kernel
import logging

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize

from sklearn import metrics
from collections import Counter
from time import time
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict

from nltk.cluster import KMeansClusterer, euclidean_distance,cosine_distance

import csv

clusters=5
categories=0

source="train_set.csv"
df=pd.read_csv(source,sep="\t")

vectorizer=CountVectorizer(stop_words='english',min_df=0)
transformer=TfidfTransformer(sublinear_tf=True)
cmatrix=vectorizer.fit_transform(df['Content'])
tfmatrix=transformer.fit_transform(cmatrix)
svd=TruncatedSVD(n_components=50)
X_train=svd.fit_transform(tfmatrix)
X_train_countvect=cmatrix


#for categories
le = LabelEncoder()
le.fit(df["Category"])
category_trans=le.transform(df["Category"])
categories=len(le.classes_)

fg=np.zeros((clusters,categories),dtype=int)

#KMeans clustering
clt=KMeansClusterer(clusters, cosine_distance, repeats=20)
result = clt.cluster(X_train, True)

print(result)

##########################################################################
fg_count=np.zeros((clusters),dtype=int)
fg_per=np.zeros((clusters,categories))

#for each cluster in result find category members

for i in range(len(result)):
	fg[result[i],category_trans[i]]=fg[result[i],category_trans[i]]+1
	fg_count[result[i]]=fg_count[result[i]] + 1

for j in range(clusters):
	for i in range(categories):
		fg_per[j,i]=format((float(fg[j,i])/fg_count[j]),'.3f')


categories=[' ', 'Business', 'Politics', 'Film', 'Football', 'Technology']
myfile=open('clustering_KMeans.csv', 'wb')
wr=csv.writer(myfile, quoting=csv.QUOTE_ALL)
wr.writerow(categories)
for i in range(clusters):
    row=[]
    row.append('Cluster'+str(i))
    for k in range(len(fg_per[i])):
    	row.append(fg_per[i][k])
    wr.writerow(row)


    