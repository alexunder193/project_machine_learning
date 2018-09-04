import pandas as pd
import gmplot
from ast import literal_eval
from haversine import haversine
from dtw import dtw
from random import choice
from string import lowercase
import numpy as np
import matplotlib.pyplot as plt


def longitudes(l):
	newlist=[]
       
	for i in l[::2]:
  		newlist.append(i)
	return newlist

def latitudes(l):
	newlist=[]
       
	for i in l[1::2]:
  		newlist.append(i)
	return newlist


#taksinomhsh listas apo tuples  kai epistrofh twn 5  prwtwn stoixeiwn ths taksinomhmenhs listas se mia nea lista
def kmost( l,n):
	l.sort(key=lambda tup: tup[0])
	l=l[0:n]
        return l



def draw(longitudes,latitudes,string):
	gmap = gmplot.GoogleMapPlotter(latitudes[0],longitudes[0], 12);
	gmap.plot(latitudes,longitudes,'green', edge_width=5);
	gmap.scatter(latitudes,longitudes, '#3B0B39', size=40, marker=False);
	gmap.scatter(latitudes,longitudes, 'k', marker=False);
	gmap.heatmap(latitudes,longitudes);
	gmap.draw(string)
	

testSet = pd.read_csv('test_set_a1.csv',converters={"Trajectory": literal_eval},sep="\t")
trainSet = pd.read_csv('train_set.csv',converters={"Trajectory": literal_eval},index_col='tripId')
trainSet=trainSet
testSet=testSet
n=3
#print (testSet);
for test in testSet['Trajectory']:		#gia kathe trajectory tou test
        
        counter=1
        print("Test trip "+str(counter)+"\n")
        distances=[]
        trajectories=[]
	testobject=[]
	string = "".join(choice(lowercase) for i in range(n))
        print("File :"+string+"\n")
	for x in test: 
		testobject.append(x[1])          #bazoume enallaks x kai y
		testobject.append(x[2])
	length=len(testobject)/2
	#print(length)
	x=np.array(testobject)                 
	x=x.reshape(length,2)                   #kanoume reshape ton pinaka me grammes to plhthos twn shmeiwn k stiles
	#print (x)                              #ta x kai y
	for i in range(0, len(trainSet)):	#gia kathe trajectory tou train
		trainobject=[]
		train=trainSet.iloc[i]['Trajectory']
                journey=trainSet.iloc[i]['journeyPatternId']
		for y in train:
			trainobject.append(y[1])
			trainobject.append(y[2])
			
		
		length=len(trainobject)/2
		y=np.array(trainobject)
		y=y.reshape(length,2)
		dist, cost, acc, path = dtw(x,y,dist=haversine)
		data=(dist,journey)
                distances.append(data)
                data=(dist,trainobject)
                trajectories.append(data)
       
        #print distances  
        
       # print trajectories
             
	distances=kmost(distances,5) 
        trajectories=kmost(trajectories,5)   		
        testlongitudes=longitudes(testobject)
	testlatitudes=latitudes(testobject)
	draw(testlongitudes,testlatitudes,string+"TEST")
        for i in range(0,len(trajectories)):
		trainlongitudes=longitudes(trajectories[i][1])
		trainlatitudes=latitudes(trajectories[i][1])
	        string_val=string+str(counter)
                draw(trainlongitudes,trainlatitudes,string_val)
                
                print("Neighbor"+str(counter)+"\n")
		print("DTW km:",trajectories[i][0])
		print(   "JP-ID",distances[i][1])
                print("\n")
                counter=int(counter)+1
	        
