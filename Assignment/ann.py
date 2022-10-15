from email import utils
import os, sys
import numpy as np
import math


from data import readDataLabels, normalize_data, train_test_split, to_categorical
from utils import accuracy_score

# Create an MLP with 8 neurons
# Input -> Hidden Layer -> Output Layer -> Output
# Neuron = f(w.x + b)
# Do forward and backward propagation

mode = 'train'      # train/test... Optional mode to avoid training incase you want to load saved model and test only.

class ANN:
    def __init__(self, num_input_features, num_hidden_units, num_outputs, hidden_unit_activation, output_activation, loss_function):
        self.num_input_features = num_input_features
        self.num_hidden_units = num_hidden_units
        self.num_outputs = num_outputs

        self.hidden_unit_activation = hidden_unit_activation
        self.output_activation = output_activation
        self.loss_function = loss_function


    def initialize_weights(self):   # TODO
        # Create and Initialize the weight matrices
        # Never initialize to all zeros. Not Cool!!!
        # Try something like uniform distribution. Do minimal research and use a cool initialization scheme.
        w1=np.zeros((self.num_input_features, self.num_hidden_units)) # creating matrice with zero value and then performing
        for i in range(0,self.num_input_features):  #uniform distribution on weights
            for j in range(0,self.num_hidden_units):
                w1[i][j]= 1/math.sqrt(self.num_hidden_units)
        w2=np.zeros((self.num_hidden_units, self.num_outputs))
        for i in range(0,self.num_hidden_units):
            for j in range(0,self.num_outputs):
                w1[i][j]= 1/math.sqrt(self.num_outputs)
        self.w1=w1
        self.w2=w2

        return w1, w2

    def forward(self,x):      # TODO
        # x = input matrix
        # hidden activation y = f(z), where z = w.x + b
        # output = g(z'), where z' =  w'.y + b'
        # Trick here is not to think in terms of one neuron at a time
        # Rather think in terms of matrices where each 'element' represents a neuron
        # and a layer operation is carried out as a matrix operation corresponding to all neurons of the layer
        y1= np.matmul(x,self.w1)
        z1=self.hidden_unit_activation(y1)
        y2=np.matmul((z1,self.w2))
        y_pred=self.output_activation(y2)
        self.y_pred =y_pred
        

    def backward(self,y_gt):     #TODO
        loss=np.zeros((len(y_gt),1))
        for i in range(len(y_gt)):
            y=max(self.y_pred[i])
            loss[i]=self.loss_function(y,y_gt[i])
        

        

    def update_params(self):    # TODO
        # Take the optimization step.
        return

    def train(self, dataset, learning_rate=0.01, num_epochs=100):
        for epoch in range(num_epochs):
            pass

    def test(self, test_dataset):
        accuracy = 0    # Test accuracy
        # Get predictions from test dataset
        # Calculate the prediction accuracy, see utils.py
        return accuracy


def main(argv):
    #ann = ANN()

    # Load dataset
    dataset = readDataLabels()      # dataset[0] = X, dataset[1] = y
    data, labels = dataset[0], dataset[1]
    # Split data into train and test split. call function in data.py
    X_train, y_train, X_test, y_test = train_test_split(data, labels,0.8)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    # call ann->train()... Once trained, try to store the model to avoid re-training everytime
    if mode == 'train':
        pass        # Call ann training code here
    else:
        # Call loading of trained model here, if using this mode (Not required, provided for convenience)
        raise NotImplementedError

    # Call ann->test().. to get accuracy in test set and print it.


if __name__ == "__main__":
    main(sys.argv)
