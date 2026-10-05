import numpy as np
from abc import abstractmethod, ABC

class Activation(ABC):
    @abstractmethod
    def forward(self, z):
        ...
        
    @abstractmethod
    def derivative(self, z):
        ...

class Sigmoid(Activation):
    def forward(self, z):
        return 1 / (1 + np.exp(-z))
    def derivative(self, z):
            return (1 / (1 + np.exp(-z))) * (1 - (1 / (1 + np.exp(-z))))
        
class ReLu(Activation):
    
    def forward(self, z):
        return (z > 0) * z
    
    def derivative(self, z):
        return (z > 0) * 1


class Layer:

    def __init__(self, n_hidden, activation_function):
        
        self.w = None
        self.b = None
        
        self.activation_function = activation_function
        self.n_hidden = n_hidden



    def forward_pass(self, a_prev):
        # input shape: (batch, features) / (samples, features)
        # weight shape: (features, hidden)
        # bias shape: (1, hidden)
        if self.w is None:
            #self.w = np.random.random((a_prev.shape[1], self.n_hidden))
            self.w = np.random.randn(a_prev.shape[1], self.n_hidden) * np.sqrt(2 / a_prev.shape[1])
            self.b = np.zeros((1, self.n_hidden))
        
        self.a_prev = a_prev
        self.z = self.a_prev @ self.w + self.b
        self.a = self.activation_function.forward(self.z)
        # a shape: samples, features
        return self.a
    
    def backward_pass(self, dL_da):
        
        # dlda shape (samples, nhidden)
        dL_dz = dL_da * self.activation_function.derivative(self.z)
        
        # shape (features, hidden)
        self.dL_dw = self.a_prev.T @ dL_dz
        # shape (1, hidden)
        self.dL_db = dL_dz.sum(axis=0, keepdims=True)
        # shape (samples, features)
        dL_da_prev = dL_dz @ self.w.T
        
        return dL_da_prev
    
    def update_weights(self, learn_rate):
        
        self.b -= self.dL_db * learn_rate
        self.w -=  self.dL_dw * learn_rate 
        
        
class MSE:
    
    def loss_calculation(self, y, y_hat):
        return np.mean(0.5 * (y_hat - y) **2)
    
    def loss_gradient(self, y, y_hat):
        return (y_hat - y) / y_hat.size


class NeuralNetwork:
    """
    Simple neural network with 1 hidden layer of flexible size
    Input: (n_samples, m_features)


    """
    def __init__(self, 
                 layers: list[Layer], 
                 learn_rate = 0.3):

        self.learn_rate = learn_rate
        self.layers = layers
        self.loss_function = MSE()


    def train(self, X, Y):
        
        for i in range(10000):
            
            layer_output = self.layers[0].forward_pass(X)
            for layer in self.layers[1:]:
                layer_output = layer.forward_pass(layer_output)
                
            loss = self.loss_function.loss_calculation(Y, layer_output)
            print(f"Epoch {i+1} Loss: {loss.round(4)}") if i % 1000 == 0 else None
            
            dL_da = self.loss_function.loss_gradient(Y, layer_output)
            for layer in self.layers[::-1]:
                dL_da = layer.backward_pass(dL_da)
                
            for layer in self.layers:
                layer.update_weights(self.learn_rate)
                
                


def main():

    x = np.array([[1, 0, 3],[4,2,2], [0,0,0], [7,0,0], [0,0,5]])
    y = np.array([[0,1,0,],[0,0,0],[1,1,1], [0,1,1], [1,1,0]])

    nn = NeuralNetwork([Layer(100, ReLu()), Layer(3,Sigmoid())])
    nn.train(x, Y=y)

main()