from email import utils
import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt


from data import readDataLabels, normalize_data, train_test_split, to_categorical
from utils import CrossEntropyLoss, ReLUActivation, SigmoidActivation, SoftmaxActivation, accuracy_score, MSELoss

# Create an MLP with 8 neurons
# Input -> Hidden Layer -> Output Layer -> Output
# Neuron = f(w.x + b)
# Do forward and backward propagation

# train/test... Optional mode to avoid training incase you want to load saved model and test only.
mode = 'train'


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
        # creating matrice with zero value and then performing

        #w1 = np.zeros((self.num_input_features, self.num_hidden_units))
        # for i in range(0, self.num_input_features):  # uniform distribution on weights
        #     for j in range(0, self.num_hidden_units):
        #         w1[i,j] = 1/math.sqrt(self.num_hidden_units)

        #w2 = np.zeros((self.num_hidden_units, self.num_outputs))
        # for i in range(0, self.num_hidden_units):
        #     for j in range(0, self.num_outputs):
        #         w2[i,j] = 1/math.sqrt(self.num_outputs)

        w1 = np.random.uniform(0,1, (self.num_input_features, self.num_hidden_units))
        b1 = np.zeros((1, self.num_hidden_units))

        print("Shape of Weights on hidden layer:-", w1.shape)
        print("Shape of biases on hidden layer", b1.shape)
        
        w2 = np.random.uniform(0, 1, (self.num_hidden_units, self.num_outputs))
        b2 = np.zeros((1, self.num_outputs))

        print("Shape of Weights on output layer:-", w2.shape)
        print("Shape of biases on output layer", b2.shape)

        self.w1 = w1
        self.w2 = w2

        self.b1 = b1
        self.b2 = b2

        self.loss_list = []
        self.accuracy_list = []

        return w1, w2, b1, b2

    def forward(self, x):      # TODO
        # x = input matrix
        # hidden activation y = f(z), where z = w.x + b
        # output = g(z'), where z' =  w'.y + b'
        # Trick here is not to think in terms of one neuron at a time
        # Rather think in terms of matrices where each 'element' represents a neuron
        # and a layer operation is carried out as a matrix operation corresponding to all neurons of the layer
        self.x = x
        self.y1 = np.dot(x, self.w1) + self.b1
        
        self.z1 = self.hidden_unit_activation.__call__(self.y1)
        
        self.y2 = np.dot(self.z1, self.w2) + self.b2
        
        self.y_pred = self.output_activation.__call__(self.y2)        
        
        return

    def backward(self, y_gt):  # TODO

        self.y_gt = y_gt
        loss = self.loss_function(self.y_pred, self.y_gt)
        self.loss_list.append(np.sum(loss))

        self.accuracy_list.append(accuracy_score(self.y_gt, self.y_pred))        
        loss_gradient = self.loss_function.grad()        
        
        grad_z = self.output_activation.__grad__()        
        y2_gradient = loss_gradient * grad_z

        w2_gradient = np.dot(np.transpose(self.z1), y2_gradient)
        b2_gradient = np.sum(y2_gradient, axis=0)
        

        z1_gradient = np.dot(y2_gradient, np.transpose(self.w2))
        
        sigmoid_gradient = self.hidden_unit_activation.__grad__()
        incoming_gradient = sigmoid_gradient * z1_gradient
        
        w1_gradient = np.dot(np.transpose(self.x), incoming_gradient)
        b1_gradient = np.sum(incoming_gradient, axis=0)
        

        self.w2_gradient = w2_gradient
        self.b2_gradient = b2_gradient

        self.w1_gradient = w1_gradient
        self.b1_gradient = b1_gradient
        
        return

    def update_params(self):    # TODO
        # Take the optimization step.

        self.w1 -= self.learning_rate * self.w1_gradient
        self.w2 -= self.learning_rate * self.w2_gradient
        self.b1 -= self.learning_rate * self.b1_gradient
        self.b2 -= self.learning_rate * self.b2_gradient
        

        return

    def train(self, dataset, labels, learning_rate=0.001, num_epochs=1000):
        self.learning_rate = learning_rate
        for epoch in range(num_epochs):
            #print(epoch)

            self.forward(dataset)
            self.backward(labels)
            self.update_params()
            #print(self.accuracy_list[epoch])
            # print(self.loss_list[epoch])
        print("Accuracy of Training set :-", self.accuracy_list[-1])
        #plt.plot(self.accuracy_list)
        

        return

    def test(self, test_dataset, test_labels):
        accuracy = 0    # Test accuracy
        # Get predictions from test dataset
        # Calculate the prediction accuracy, see utils.py

        y1 = np.dot(test_dataset, self.w1) + self.b1
        z1 = self.hidden_unit_activation.__call__(y1)
        y2 = np.dot(z1, self.w2) + self.b2
        y_pred = self.output_activation.__call__(y2)

        accuracy = accuracy_score(test_labels, y_pred)

        return accuracy


def main(argv):

    # Load dataset
    dataset = readDataLabels()      # dataset[0] = X, dataset[1] = y
    data, labels = dataset[0], dataset[1]
    # Split data into train and test split. call function in data.py
    labels = to_categorical(labels)
    data = normalize_data(data)
    
    dataset = data, labels
    X_train, y_train, X_test, y_test = train_test_split(data, labels, 0.8)
    print('Shape of X_train is :', X_train.shape)
    print('Shape of y_train is :', y_train.shape)
    print('Shape of X_test is :', X_test.shape)
    print('Shape of y_test is :', y_test.shape)
    sigmoid = SigmoidActivation()
    softmax = SoftmaxActivation()
    loss_function = CrossEntropyLoss()
    # call ann->train()... Once trained, try to store the model to avoid re-training everytime
    if mode == 'train':
        # Call ann training code here
        ann = ANN(X_train.shape[1], 16, 10, sigmoid, softmax, loss_function)
        w1, w2, b1, b2 = ann.initialize_weights()
        ann.train(X_train, y_train)
        

    else:
        # Call loading of trained model here, if using this mode (Not required, provided for convenience)
        raise NotImplementedError

    # Call ann->test().. to get accuracy in test set and print it.
    print("Accuracy of Test set :-", ann.test(X_test, y_test))


if __name__ == "__main__":
    main(sys.argv)
