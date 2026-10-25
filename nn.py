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
    
class Identity(Activation):
    
    def forward(self, z):
        return z
    
    def derivative(self, z):
        return 1


class Softmax(Activation):
    
    def forward(self, logits):
        
        y_hat_shifted = logits - logits.max(axis=1, keepdims=True)
        s = np.sum(np.exp(y_hat_shifted), axis=1, keepdims=True)
        return np.exp(y_hat_shifted) / s
        
    
class MSE:
    
    def loss_calculation(self, y_hat, y):
        return np.mean(0.5 * (y_hat - y) **2)
    
    def loss_gradient(self, y_hat, y):
        return (y_hat - y) / y_hat.size
    
class CrossEntropy:
    
    def loss_calculation(self, logits, y):
        y_hat_shifted = logits - logits.max(axis=1, keepdims=True)
        s = np.sum(np.exp(y_hat_shifted), axis=1, keepdims=True)
        sample_loss = - np.sum(y * (y_hat_shifted - np.log(s)), axis=1)
        
        return sample_loss.mean()
    
    def loss_gradient(self, logits, y):
        
        y_hat_shifted = logits - logits.max(axis=1, keepdims=True)
        s = np.sum(np.exp(y_hat_shifted), axis=1, keepdims=True)
        a = np.exp(y_hat_shifted) / s
        
        return (a - y) / y.shape[0]
        
        


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
        
        



class NeuralNetwork:
    """
    Simple neural network with n hidden layer of flexible size
    Input: (n_samples, m_features)
    """
    def __init__(self, 
                 layers: list[Layer],
                 loss_function):

        self.layers = layers
        self.loss_function = loss_function
        
    def accuracy(self, y_hat, Y):
        return (y_hat.argmax(axis=1) == Y.argmax(axis=1)).sum() / Y.shape[0]
        

    def train(self, X, Y, epochs=1, batch_size=100, learn_rate = 0.05, X_val = None, Y_val = None):
        
        layer_output = None
        self.epoch_losses = []
        self.epoch_losses_val = []
        self.epoch_accu = []
        self.epoch_accu_val = []  
        for i in range(epochs):
            permutation = np.random.permutation(X.shape[0])
            loss = 0

            for indices in range(0, X.shape[0], batch_size):
                batch_indices = permutation[indices:indices+batch_size]
                X_batch = X[batch_indices]
                Y_batch = Y[batch_indices]
                
                layer_output = self.forward(X_batch)
                    
                loss += self.loss(layer_output, Y_batch) * batch_indices.shape[0]
                
                self.backward(layer_output, Y_batch)
                    
                for layer in self.layers:
                    layer.update_weights(learn_rate)
            
            epoch_loss_batch_avg = loss / permutation.shape[0]

            print(f"Average batch loss in epoch {i}: {epoch_loss_batch_avg.round(4)}") 
            
            if X_val is not None and Y_val is not None:
                layer_output_val = self.forward(X_val)  
                self.epoch_losses_val.append(self.loss(layer_output_val, Y_val))
                self.epoch_accu_val.append(self.accuracy(layer_output_val, Y_val))
                
                layer_output = self.forward(X)  
                self.epoch_losses.append(self.loss(layer_output, Y))
                self.epoch_accu.append(self.accuracy(layer_output, Y))

        return 
            
    def forward(self, X) -> np.ndarray:
        """
        Forward pass through the network layers.
        Parameters:
            X: Input data.
        Returns:
            np.ndarray: Network predictions
        """
        layer_output = self.layers[0].forward_pass(X)
        for layer in self.layers[1:]:
            layer_output = layer.forward_pass(layer_output)
        return layer_output
    
    def backward(self, layer_output, Y):
            
        dL_da = self.loss_function.loss_gradient(layer_output, Y)
        for layer in self.layers[::-1]:
            dL_da = layer.backward_pass(dL_da)
            
    def loss(self, layer_output, Y):
        return self.loss_function.loss_calculation(layer_output, Y)

def train_val_split(X, Y, train_split):
    permutation = np.random.permutation(X.shape[0])
    pivot = round(X.shape[0] * train_split)
    train_indices = permutation[:pivot]
    val_indices = permutation[pivot:]
    
    return X[train_indices], Y[train_indices], X[val_indices], Y[val_indices]
    

def main():

    x = np.array([[1,55, 3],[4,100,22], [11,33,0], [7,0,0], [0,0,224]])
    y = np.array([[0,1,0,],[0,0,0],[1,1,1], [0,1,1], [1,1,0]])

    nn = NeuralNetwork([Layer(3, ReLu()), Layer(3,Sigmoid())], MSE())
    #nn.train(x, Y=y, epochs=1)
    print(train_val_split(x, y, 0.7, 0))

if __name__== "__main__":
    main()
