import os, sys
import numpy as np
from sklearn import datasets

#from sklearn.model_selection import train_test_split


#import libraries as needed

def readDataLabels(): 
	#read in the data and the labels to feed into the ANN
	data = datasets.load_digits()
	X = data.data
	y = data.target

	return X,y

def to_categorical(y):
	
	#Convert the nominal y values tocategorical

	return y
	
def train_test_split(data,labels,n=0.8): #TODO

	#split data in training and testing sets
	#X_train, X_test = data.load_data(n=0.8)
	data_len=data.shape
	labels_len=labels.shape
	#print(data_len, labels_len)
	X_train=np.zeros((int(data_len[0]*0.8),data_len[1]))
	for i in range(0,int(data_len[0]*0.8)):
		for j in range(0,int(data_len[1]*0.8)):
			X_train[i][j]=data[i][j]
	y_train=np.zeros((int(labels_len[0]*0.8),1))
	for i in range(0,int(labels_len[0]*0.8)):
		y_train[i]=labels[i]
	X_test=np.zeros((int(data_len[0]*0.2)+1,data_len[1]))
	k,l=0,0
	for i in range(0, int(data_len[0]*0.2)+1):
		l=0
		for j in range(0, int(data_len[1]*0.2)):
			X_test[k][l]=data[i][j]
			l+=1
		k+=1
	y_test=np.zeros((int(labels_len[0]*0.2)+1,1))
	k=0
	for i in range(0, int(labels_len[0]*0.2)+1):
		y_test[k]=labels[i]
		k+=1
		
	print(X_train.shape)
	print(y_train.shape)
	print(X_test.shape)
	print(y_test.shape)
	print(X_train)
	print(y_train)
	#for i in y_train:
	#	print(i)

	return X_train, y_train, X_test, y_test

def normalize_data(data): #TODO

	# normalize/standardize the data
	data=data/255.0

	return data

data, labels = readDataLabels()
#print(data)
X_train, X_test, y_train, y_test = train_test_split(data, labels,n=0.8)
#print(X_train.shape)
#print(y_train.shape) --> Tested the split by using scikit library and matches with code mention in the split function
#print(X_test.shape)
#print(y_test.shape)
#data=normalize_data(data)
#train_test_split(data,labels)


