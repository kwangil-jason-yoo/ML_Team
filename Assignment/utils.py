import numpy as np
import math


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
        loss = None
        #CEL= - GT x log(prediction)
        loss = -y_gt * math.log(y_pred,10) # 10 is base of logarithm
        return loss

    def grad(self):
        # TODO: Calculate Gradients for back propagation
        gradient = 0.5 * (self.current_gt / self.current_prediction)
        self.current_prediction = None
        self.current_gt = None
        return gradient


class SoftmaxActivation:    # TODO: Make this work!!!
    def __init__(self):
        self.y = None
        pass

    def __call__(self, y):
        # TODO: Calculate Activation Function
        #f(y)= exp(y)/Sum(exp(y))
        self.y=y
        y_length= y.shape
        z=np.zeros((y_length[0],y_length[1]))
        for i in range(0,y_length[0]):
            exp_y= np.exp(y[i])
            z[i]=exp_y/exp_y.sum()
        #sum_exp=0
        #z=[]
        #for i in range(0,len(y)):
        #    sum_exp+=np.exp(y[i])
        #for i in range(0,len(y)):
        #    z.append(np.exp(y[i])/sum_exp)
        return z


    def __grad__(self):
        # TODO: Calculate Gradients.. Remember this is calculated w.r.t. input to the function -> dy/dz
        pass


class SigmoidActivation:    # TODO: Make this work!!!
    def __init__(self):
        self.y = None
        pass

    def __call__(self, y):
        # TODO: Calculate Activation Function
        #f(y)=exp(y)/(exp(y)+1)
        self.y =y
        y_length= y.shape
        z=np.zeros((y_length[0],y_length[1]))
        for i in range(0,y_length[0]):
            for j in range(0,y_length[1]):
                z[i][j]= np.exp(y[i][j])/(np.exp(y[i][j]) + 1)
        return z

    def __grad__(self):
        # TODO: Calculate Gradients.. Remember this is calculated w.r.t. input to the function -> dy/dz
        #gradient = exp(-y) / (1+exp(-y))^2
        y_length= self.y.shape
        gradient=np.zeros((y_length[0],y_length[1]))
        for i in range(0,y_length[0]):
            for j in range(0,y_length[1]):
                gradient[i][j] = np.exp(-self.y[i][j]) / (np.power((1 + np.exp(-self.y[i][j])),2))
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
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy
