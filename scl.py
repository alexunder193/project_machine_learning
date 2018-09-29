import pandas as pd
import csv
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS,CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import nltk
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk  import sent_tokenize


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

train="train_set.csv"
test="test_set.csv"
name=['Id','Category']
df=pd.read_csv(train,sep="\t")
df_test=pd.read_csv(test,sep="\t")

#removing punctuation
df['Content'] = df['Content'].str.replace('[^\w\s]','')

X_train=df['Content']
X_test=df_test['Content']
    
column0=df_test['Id']

le=LabelEncoder()
Y_train=le.fit_transform(df['Category'])

#vect=TfidfVectorizer(stop_words='english',min_df=0)
svd=TruncatedSVD(n_components=10,random_state=42)
stemmed_count_vect = StemmedCountVectorizer(stop_words=ENGLISH_STOP_WORDS)
clf=RandomForestClassifier()
#clf=KNeighborsClassifier()
pipeline = Pipeline([('vect',stemmed_count_vect),('tfidf', TfidfTransformer()),('svd',svd),('clf', clf)])
    #Pipeline Fit
pipeline.fit(X_train,Y_train)
    #Predict the train set
predicted=pipeline.predict(X_test)

column1=le.inverse_transform(predicted)

myfile = open('testSet_categories.csv', 'wb')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
wr.writerow(name)
for i in range(len(column0)):
    row=[]
    row.append(column0[i]) 
    row.append(column1[i])
    wr.writerow(row)

