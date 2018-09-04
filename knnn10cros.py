import csv
import random
import math
import operator
import pandas as pd
import numpy as np
from ast import literal_eval
from dtw import dtw
from haversine import haversine



def loadDataset(filename, split):
    
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



             
    

def getNeighbors(trainingSet, testInstance, k):
    distances=[]
    testtrajectory=[]
    for i in range(len(testInstance[2])):
        testtrajectory.append(testInstance[2][i][0])
        testtrajectory.append(testInstance[2][i][1])
    test=np.array(testtrajectory)                 
    test=test.reshape(len(testInstance[2]),2)
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
    distances.sort(key = operator.itemgetter(1))
    neighbors = []
    print("Eksetasa kampuli testset")
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
    print ("Swstes provlepseis:")
    print (correct)
    return (correct/float(len(testSet))) * 100.0
                
def main():
    idset=[]
    journeyset=[]
    twodimlist=[]
    split = 0.90
    idset,journeyset,twodimlist=loadDataset('train_set.csv',split)
    trainset=[]
    testset=[]
    for i in range(100):
        tup=(idset[i],journeyset[i],twodimlist[i])
        if random.random() < split:
            trainset.append(tup)
        else:
            testset.append(tup)
    predictions=[]
    k = 5
    print("Megethos Dataset:")
    print(len(trainset))
    print("Megethos Testset:")
    print(len(testset))
    for x in range(len(testset)):
        neighbors = getNeighbors(trainset,testset[x],k)
        result = getResponse(neighbors)
        predictions.append(result)
    accuracy = getAccuracy(testset, predictions)
    print("Accuracy:")
    print(accuracy)
    
main()
