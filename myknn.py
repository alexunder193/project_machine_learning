
from sklearn import cross_validation,preprocessing
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS,CountVectorizer,TfidfTransformer,TfidfVectorizer
from operator import itemgetter
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
import numpy as np
import math
import pandas as pd
from collections import Counter

#we compute the euclideian distance of two points 
def get_distance(data1, data2):
    coordinates = zip(data1, data2)
    diffs_squared_distance = [pow(a - b, 2) for (a, b) in coordinates]  #(a-b)^2
    return math.sqrt(sum(diffs_squared_distance))    #square root of sum 

 
def get_neighbours(training_set, test_instance, k):
    distances = [_get_tuple_distance(train, test_instance) for train in training_set]
    # the calculated distance between training_instance and test_instance is stored in index 1
    sorted_dist = sorted(distances, key=itemgetter(1))
    #  training instances are stored in index 0
    sorted_training_instances = [tuple[0] for tuple in sorted_dist]
    # select k first elements,aka training points with smallenst distance from test instance
    return sorted_training_instances[:k]
#computes the k nearest points of a test instance

def get_neighbours(training_set, test_instance, k):
    #computes pairwise distances between test instance and all the training instances
    distances = [_get_tuple_distance(train, test_instance) for train in training_set]
    # the calculated distance between training and test_instance
     #is stored in index 1

    #sorts the pairwise instances
    sorted_dist = sorted(distances, key=itemgetter(1))
    # take the  training instances from index 0 
    sorted_training = [tuple[0] for tuple in sorted_dist]
    # select the first k elements,aka the k training instances with smallest distance
    return sorted_training[:k]
 

#this function is called to create a tuple with training instance stored in zero index and distance between training and test instance in index 1
def _get_tuple_distance(training_instance, test_instance):
    return (training_instance, get_distance(test_instance, training_instance[0]))
 
#this function does majority voting.That means , that is sees the category of the k -nearest neighbors and predicts the majority of them as the category of the test instance
def get_majority_vote(neighbours):
    predictions = [neighbour[1] for neighbour in neighbours]
    count = Counter(predictions)
    return count.most_common()[0][0] 
 
# setting up main executable method
def main():
     
    # load the data and create the training and test sets
    # random_state = 1 is just a seed to permit reproducibility of the train/test split
    df=pd.read_csv('train_set.csv' ,sep="\t")
    df.drop(['Id'],1,inplace=True)   #getting rid of unnecessary data as id is not critical for predictions
    train_data = pd.read_csv('train_set.csv', sep="\t")
    train_data = train_data[0:100]
    test_data = pd.read_csv('test_set.csv', sep="\t")
    test_data = test_data[0:100]
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
    X = svd_transformer.fit_transform(train_data['Content'])
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=1)
 
    # reformat train/test datasets for convenience
    train = np.array(zip(X_train,y_train))
    test = np.array(zip(X_test, y_test))
    # generate predictions
    predictions = []
 
    # let's arbitrarily set k equal to 5, meaning that to predict the class of new instances,
    k = 5
  # for each instance in the test set, get nearest neighbours and majority vote on predicted class
    for x in range(len(X_test)):
 
          #  print 'Classifying test instance number ' + str(x) + ":",
            neighbours = get_neighbours(training_set=train, test_instance=test[x][0], k=5)
            majority_vote = get_majority_vote(neighbours)
            predictions.append(majority_vote)
            #print 'Predicted label=' + str(majority_vote) + ', Actual label=' + str(test[x][1])
 
    # summarize performance of the classification
    print 'The overall accuracy of the model is: ' + str(accuracy_score(y_test, predictions)) + "\n"
    report = classification_report(y_test, predictions, target_names=list(le.classes_))
    print 'A detailed classification report: \n\n' + report
 
if __name__ == "__main__":
    main()