import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder

from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

train="train_set.csv"
test="test_set.csv"

df=pd.read_csv(train,sep="\t")
df_test=pd.read_csv(test,sep="\t")

X_train=df['Content']
X_test=df_test['Content']
    
column0=df_test['Id']

le=LabelEncoder()
Y_train=le.fit_transform(df['Category'])

vect=TfidfVectorizer(stop_words='english',min_df=0)
svd=TruncatedSVD(n_components=10,random_state=42)
clf=KNeighborsClassifier()

pipeline = Pipeline([('vect', vect),('svd',svd),('clf', clf)])
    #Pipeline Fit
pipeline.fit(X_train,Y_train)
    #Predict the train set
predicted=pipeline.predict(X_test)

column1=le.inverse_transform(predicted)

myfile = open('testSet_categories.csv', 'wb')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
for i in range(len(column0)):
    row=[]
    row.append(column0[i]) 
    row.append(column1[i])
    wr.writerow(row)

