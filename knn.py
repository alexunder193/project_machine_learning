import csv
import random
import math
import operator
import pandas as pd
import numpy as np
from ast import literal_eval
from dtw import dtw
from haversine import haversine



def loadDataset(filename):
    
    idset=[]
    journeyset=[]
    trainSet = pd.read_csv(filename,converters={"Trajectory": literal_eval,"tripId": literal_eval, "journeyPatternId":str})
    trainSet=trainSet[350:450]
    for identity in trainSet['tripId']:
        idset.append(identity)
    for journey in trainSet['journeyPatternId']:
        journeyset.append(journey)
    twodimlist=[]
    for test in trainSet['Trajectory']:
        trajectory=[]
        for x in test:
            tup=(x[1],x[2])
            trajectory.append(tup)
        twodimlist.append(trajectory)
    return idset,journeyset,twodimlist


def  loadTestset(filename):
    testSet = pd.read_csv('test_set_a2.csv',converters={"Trajectory": literal_eval},sep="\t")
    twodimlist=[]
    for test in testSet['Trajectory']:
        trajectory=[]
        for x in test:
            tup=(x[1],x[2])
            trajectory.append(tup)
        twodimlist.append(trajectory)
    return twodimlist
    

def getNeighbors(trainingSet, testInstance, k):
    distances=[]
    testtrajectory=[]
    #print("mikos grammis")
    #print(len(testInstance[2]))
    for i in range(len(testInstance)):
        testtrajectory.append(testInstance[i][0])
        testtrajectory.append(testInstance[i][1])
    test=np.array(testtrajectory)                 
    test=test.reshape(len(testInstance),2)
    for y in range(len(trainingSet)):
        traintrajectory=[]
        for z in range(len(trainingSet[y][2])):
            traintrajectory.append(trainingSet[y][2][z][0])
            traintrajectory.append(trainingSet[y][2][z][1])
        train=np.array(traintrajectory)
        train=train.reshape(len(trainingSet[y][2]),2)
        #print(y)
        distance,cost,acc,path=dtw(train,test,dist=haversine)
        distances.append((trainingSet[y],distance))
    print("Eksetasa kampuli testset") 
    distances.sort(key = operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    # Creating a list with all the possible neighbors
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    #print(classVotes)
    sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][1] == predictions[x]:
            correct += 1
    print (len(testSet))
    print (correct)
    return (correct/float(len(testSet))) * 100.0
                
def main():
    idset=[]
    journeyset=[]
    twodimlist=[]
    testlist=[]
    idset,journeyset,twodimlist=loadDataset('train_set.csv')
    testlist=loadTestset('test_set_a2.csv')
    trainset=[]
    testset=[]
    for i in range(100):
        tup=(idset[i],journeyset[i],twodimlist[i])
        trainset.append(tup)
    predictions=[]
    k = 5
    print("Megethos Dataset:")
    print(len(trainset))
    print("Megethos Testset:")
    print(len(testlist))
    for x in range(len(testlist)):
        neighbors = getNeighbors(trainset,testlist[x],k)
        result = getResponse(neighbors)
        predictions.append(result)
    #print(predictions)    
    name=['Test_Trip_ID','Predicted_JourneyPatternID']
    lista1=[1,2,3,4,5]
    myfile = open('testSet_JourneyPatternIDs.csv', 'wb')
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(name)
    for i in range(len(lista1)):
        row=[]
        row.append(lista1[i]) 
        row.append(predictions[i])
        wr.writerow(row)
        
main()
