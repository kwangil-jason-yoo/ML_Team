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
        loss=np.zeros((y_gt.shape[0],1))
        for i in range(0,y_gt.shape[0]):
            sum=0
            for j in range(y_gt.shape[1]):
                sum+=y_gt[i][j] * math.log(y_pred[i][j],2)
            loss[i][1]=-1*sum
        
        return loss

    def grad(self):
        # TODO: Calculate Gradients for back propagation
        loss_gradient = np.zeros((self.current_gt.shape[0],1))
        for i in range(0,self.current_gt.shape[0]):
            sum=0
            for j in range(self.current_gtt.shape[1]):
                sum+=self.current_gt[i][j] / self.current_prediction[i][j]
            loss_gradient[i][1]=-1*sum
        return loss_gradient


class SoftmaxActivation:    # TODO: Make this work!!!
    def __init__(self):
        self.y = None
        pass

    def __call__(self, y):
        # TODO: Calculate Activation Function
        #f(y)= exp(y)/Sum(exp(y))
        self.y=y
        z=np.zeros((y.shape[0],y.shape[1]))
        for i in range(0,y.shape[0]):
            exp_sum_y=sum(np.exp(y[i]))
            for j in range(0,y.shape[1]):
                exp_y= np.exp(y[i][j])
                z[i][j]=exp_y/exp_sum_y
        self.z_soft=z

        return z


    def __grad__(self):
        # TODO: Calculate Gradients.. Remember this is calculated w.r.t. input to the function -> dy/dz
        z=self.y
        grad_z=np.zeros((z.shape[0],z.shape[1]))
        for i in range(0, z.shape[0]):
            for j in range(0, z.shape[1]):
                if i == j:
                    grad_z[i][j]= z[i][j] * (1-z[i][j])
                elif i != j:
                    grad_z[i][j]= -z[i][j] * z[i+1][j]


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
