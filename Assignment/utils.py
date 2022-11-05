from cmath import isnan
import numpy as np
import math
import sys


class MSELoss:      # For Reference
    def __init__(self):
        # Buffers to store intermediate results.
        self.current_prediction = None
        self.current_gt = None
        pass

    def __call__(self, y_pred, y_gt):
        self.current_prediction = y_pred
        self.current_gt = y_gt

        # MSE = 0.5 x (GT - Prediction)^2
        loss = 0.5 * np.power((y_gt - y_pred), 2)
        print(loss)
        return loss

    def grad(self):
        # Derived by calculating dL/dy_pred
        gradient = -1 * (self.current_gt - self.current_prediction)

        # We are creating and emptying buffers to emulate computation graphs in
        # Modern ML frameworks such as Tensorflow and Pytorch. It is not required.
        self.current_prediction = None
        self.current_gt = None

        return gradient


class CrossEntropyLoss:     # TODO: Make this work!!!
    def __init__(self):
        # Buffers to store intermediate results.
        self.current_prediction = None
        self.current_gt = None
        pass

    def __call__(self, y_pred, y_gt):
        # TODO: Calculate Loss Function

        self.current_prediction = y_pred
        self.current_gt = y_gt
        
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = - y_gt * np.log(y_pred) - (1 - y_gt) * np.log(1 - y_pred)
        
        return loss

    def grad(self):
        # TODO: Calculate Gradients for back propagation
        
        self.current_prediction = np.clip(self.current_prediction, 1e-15, 1 - 1e-15)
        loss_gradient = - (self.current_gt / self.current_prediction) + (1 - self.current_gt) / (1 - self.current_prediction)
        self.current_prediction = None
        self.current_gt = None
    

        return loss_gradient


class SoftmaxActivation:    # TODO: Make this work!!!
    def __init__(self):
        self.y = None
        pass

    def __call__(self, y):
        # TODO: Calculate Activation Function
        #f(y)= exp(y)/Sum(exp(y))
        self.y=y
        exp_y=np.exp(y- np.max(y, axis=1, keepdims=True))
        exp_sum = np.sum(exp_y, axis=1, keepdims=True)
        z = exp_y /exp_sum
        self.z_soft=z

        return z


    def __grad__(self):
        # TODO: Calculate Gradients.. Remember this is calculated w.r.t. input to the function -> dy/dz
        y=self.z_soft
        grad_z = y * (1-y)
        self.grad_z=grad_z
        return grad_z


class SigmoidActivation:    # TODO: Make this work!!!
    def __init__(self):
        self.y = None
        pass

    def __call__(self, y):
        # TODO: Calculate Activation Function
        #f(y)=exp(y)/(exp(y)+1)
        self.y =y
        #y_length= y.shape
        #z=np.zeros((y_length[0],y_length[1]))
        #for i in range(0,y_length[0]):
        #    for j in range(0,y_length[1]):
        #        z[i][j]= np.exp(y[i][j])/(np.exp(y[i][j]) + 1)
        z = 1 / (1 + np.exp(-y))
        
        self.z = z
        return z

    def __grad__(self):
        # TODO: Calculate Gradients.. Remember this is calculated w.r.t. input to the function -> dy/dz
        
        y_length= self.y.shape
        #gradient=np.zeros((y_length[0],y_length[1]))
        #for i in range(0,y_length[0]):
        #    for j in range(0,y_length[1]):
        #        gradient[i][j] = np.exp(-self.y[i][j]) / (np.power((1 + np.exp(-self.y[i][j])),2))
        gradient = (1 / (1 + np.exp(-self.y))) * (1 - (1 / (1 + np.exp(-self.y))))
        self.gradient=gradient
        return gradient


class ReLUActivation:
    def __init__(self):
        self.z = None
        pass

    def __call__(self, z):
        # y = f(z) = max(z, 0) -> Refer to the computational model of an Artificial Neuron
        self.z = z
        y = np.maximum(z, 0)
        return y

    def __grad__(self):
        # dy/dz = 1 if z was > 0 or dy/dz = 0 if z was <= 0
        gradient = np.where(self.z > 0, 1, 0)
        return gradient


def accuracy_score(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)
    accuracy = np.sum(y_true == y_pred, axis=0) / y_true.shape[0]
    accuracy*=100
    return accuracy
