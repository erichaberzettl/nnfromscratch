# Neural Network from Scratch

A small neural network written in plain NumPy. I built it to understand how backpropagation and training actually work and tested it on MNIST handwritten digits.

## What's in here

- `nn.py` – the network itself
  - `Layer`: fully connected layer with forward and backward pass (weights are initialised with He initialisation)
  - Activations: `ReLu`, `Sigmoid`, `Identity`
  - Losses: `MSE` and `CrossEntropy` (the softmax is built into the cross entropy, so the last layer just outputs raw logits)
  - `NeuralNetwork`: stacks any number of layers, trains with mini-batch gradient descent and tracks loss and accuracy per epoch
  - `train_val_split` helper to split data into training and validation set
- `nn_test.py` – gradient checking. Compares the backprop gradients to numerical gradients to make sure the maths is right.
- `mnist.ipynb` – training and evaluating the network on MNIST

## Example

```
nn = NeuralNetwork([Layer(64, ReLu()),
                    Layer(64, ReLu()),
                    Layer(10, Identity())],
                   loss_function=CrossEntropy())

nn.train(X_train, Y_train, epochs=50, batch_size=500, learn_rate=0.07,
         X_val=X_val, Y_val=Y_val)
```

## Results on MNIST

Two hidden layers with 64 neurons each, 50 epochs, batch size 500, learning rate 0.07:

| Set | Accuracy |
|---|---|
| Train | ~98.6% |
| Validation | ~96.8% |
| Test | ~96.8% |

The gap between train and validation accuracy shows a bit of overfitting, which I haven't tried to fix.

## Running it

You need `numpy`, `pandas` and `matplotlib`. The notebook expects `mnist_train.csv` and `mnist_test.csv` (label in the first column, then the 784 pixel values) in the same folder. I found the files [here](https://pjreddie.com/projects/mnist-in-csv/).

To run the gradient check:

```python
from nn_test import *
gradient_test()
```

## How I made this

I wrote all the code myself. I learned the theory from YouTube videos on neural networks and backpropagation, and I also worked through the project with Claude as a tutor, mostly to ask questions, get hints and have my reasoning checked when I got stuck. The implementation is my own.