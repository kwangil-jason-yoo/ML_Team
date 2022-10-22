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
	uniques=np.unique(y)
	y_cat=np.zeros((y.shape[0], np.amax(y)+1))
	for i in range(uniques.shape[0]):
		for j in range(y.shape[0]):
			if uniques[i]==y[j]:
				y_cat[j,i]=1
	

	return y_cat
	
def train_test_split(data,labels,n): #TODO

	#split data in training and testing sets
	#X_train, X_test = data.load_data(n=0.8)
	num_train_samples= int(labels.shape[0]*n)
	X_train = data[:num_train_samples]
	y_train = labels[:num_train_samples]
	X_test = data[num_train_samples:]
	y_test = labels[num_train_samples:]
	
	

	return X_train, y_train, X_test, y_test

def normalize_data(data): #TODO

	# normalize/standardize the data
	data_norm = (data - np.amin(data))/(np.amax(data) - np.amin(data))
	return data_norm

data, labels = readDataLabels()
print(data.shape, labels.shape)





