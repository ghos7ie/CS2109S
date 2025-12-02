# RUN THIS CELL FIRST
import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
from numpy import allclose, isclose

from collections.abc import Callable

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Define a linear layer using nn.Module
class LinearLayer(nn.Module):
    """
    Linear layer as a subclass of `nn.Module`.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.randn(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight) + self.bias
    
class SineActivation(nn.Module):
    """
    Sine activation layer.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)

class Model(nn.Module):
    """
    Neural network created using `LinearLayer` and `SineActivation`.
    """
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super(Model, self).__init__()
        self.l1 = LinearLayer(input_size, hidden_size)
        self.act = SineActivation()
        self.l2 = LinearLayer(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.act(x)
        x = self.l2(x)
        return x
    
input_size = 1
hidden_size = 1
num_classes = 1

model = Model(input_size, hidden_size, num_classes)

x = torch.tensor([[1.0]])
output = model(x)
print("Original value: ", x)
print("Value after being processed by Model: ", output)


class Squared(nn.Module):
    """
    Module that returns x**2.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.x = x
        return x**2

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        grad_input = 2 * self.x * grad_output
        return grad_input

x_sample = torch.linspace(-2, 2, 100)
sigmoid_output = nn.Sigmoid()(x_sample).detach().numpy()
tanh_output = nn.Tanh()(x_sample).detach().numpy()
relu_output = nn.ReLU()(x_sample).detach().numpy()

f = plt.figure()
f.set_figwidth(6)
f.set_figheight(6)
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title("Input: 100 x-values between -2 to 2 \n\n Output: Corresponding y-values after passed through each activation function\n", fontsize=16)
plt.axvline(x=0, color='r', linestyle='dashed')
plt.axhline(y=0, color='r', linestyle='dashed')
plt.plot(x_sample, sigmoid_output)
plt.plot(x_sample, tanh_output)
plt.plot(x_sample, relu_output)
plt.legend(["","","Sigmoid Output", "Tanh Output", "ReLU Output"])
plt.show()

# DEMO
class MyFirstNeuralNet(nn.Module):
    def __init__(self): # set the arguments you'd need
        super().__init__()
        self.l1 = nn.Linear(1, 2) # bias included by default
        self.l2 = nn.Linear(2, 1) # bias included by default
        self.relu = nn.ReLU()
 
    # Task 1.1: Forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass to process input through two linear layers and ReLU activation function.

        Parameters
        ----------
        x : A tensor of of shape (n, 1) where n is the number of training instances

        Returns
        -------
            Tensor of shape (n, 1)
        '''
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """


# Run to check
x_sample = torch.linspace(-2, 2, 5).reshape(-1, 1)

model = MyFirstNeuralNet()

state_dict = OrderedDict([
    ('l1.weight', torch.tensor([[1.],[-1.]])),
    ('l1.bias',   torch.tensor([-1., 1.])),
    ('l2.weight', torch.tensor([[1., 1.]])),
    ('l2.bias',   torch.tensor([0.]))
])

model.load_state_dict(state_dict)

student1 = model.forward(x_sample).detach().numpy()
output1 = [[3.], [2.], [1.], [0.], [1.]]

assert allclose(student1, output1, atol=1e-5)


x = torch.tensor([1.0], requires_grad=True)

#Loss function
y = x ** 2 + 2 * x

# Define an optimizer, pass it our tensor x to update
optimiser = torch.optim.SGD([x], lr=0.1)

# Perform backpropagation
y.backward()

print("Value of x before it is updated by optimiser: ", x)
print("Gradient stored in x after backpropagation: ", x.grad)

# Call the step function on the optimizer to update weight
optimiser.step()

#Weight update, x = x - lr * x.grad = 1.0 - 0.1 * 4.0 = 0.60
print("Value of x after it is updated by optimiser: ", x)

# Set gradient of weight to zero
optimiser.zero_grad()
print("Gradient stored in x after zero_grad is called: ", x.grad)

torch.manual_seed(6) # Set seed to some fixed value

epochs = 10000

model = MyFirstNeuralNet()
# the optimizer controls the learning rate
optimiser = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0)
loss_fn = nn.MSELoss()

x = torch.linspace(-10, 10, 1000).reshape(-1, 1)
y = torch.abs(x-1)

print('Epoch', 'Loss', '\n-----', '----', sep='\t')
for i in range(1, epochs+1):
    # reset gradients to 0
    optimiser.zero_grad()
    # get predictions
    y_pred = model(x)
    # compute loss
    loss = loss_fn(y_pred, y)
    # backpropagate
    loss.backward()
    # update the model weights
    optimiser.step()

    if i % 1000 == 0:
        print (f"{i:5d}", loss.item(), sep='\t')

y_pred = model(x)
plt.plot(x, y, linestyle='solid', label='|x-1|')
plt.plot(x, y_pred.detach().numpy(), linestyle='dashed', label='perceptron')
plt.axis('equal')
plt.title('Fit NN on y=|x-1| function')
plt.legend()
plt.show()

# Run to view your model weights
print("--- Submit the OrderedDict below ---")
print(model.state_dict())

# DO NOT REMOVE THIS CELL – THIS DOWNLOADS THE MNIST DATASET
# RUN THIS CELL BEFORE YOU RUN THE REST OF THE CELLS BELOW
from torchvision import datasets

# This downloads the MNIST datasets ~63MB
mnist_train = datasets.MNIST("./", train=True, download=True)
mnist_test  = datasets.MNIST("./", train=False, download=True)

x_train = mnist_train.data.reshape(-1, 784) / 255
y_train = mnist_train.targets
    
x_test = mnist_test.data.reshape(-1, 784) / 255
y_test = mnist_test.targets

### Task 1.1 - Define the model architecture and implement the forward pass

class DigitNet(nn.Module):
    def __init__(self, input_dimensions: int, num_classes: int): # set the arguments you'd need
        super().__init__()
        """
        YOUR CODE HERE
        - DO NOT hardcode the input_dimensions, use the parameter in the function
        - Your network should work for any input and output size 
        - Create the 3 layers (and a ReLU layer) using the torch.nn layers API
        """
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the network.
        
        Parameters
        ----------
        x : Input tensor (batch size is the entire dataset)

        Returns
        -------
            The output of the entire 3-layer model.
        """
        
        """
        YOUR CODE
        
        - Pass the inputs through the sequence of layers
        - Run the final output through the Softmax function on the right dimension!
        """
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """

def test_task_1_1():
    model = DigitNet(784, 10)
    assert [layer.detach().numpy().shape for name, layer in model.named_parameters()] \
            == [(512, 784), (512,), (128, 512), (128,), (10, 128), (10,)]

### Task 1.2 - Training Loop

def train_model(x_train: torch.Tensor, y_train: torch.Tensor, epochs: int = 20):
    """
    Trains the model for 20 epochs/iterations
    
    Parameters
    ----------
        x_train : A tensor of training features of shape (60000, 784)
        y_train : A tensor of training labels of shape (60000, 1)
        epochs  : Number of epochs, default of 20
        
    Returns
    -------
        The final model 
    """
    model = DigitNet(784, 10)

    optimiser = torch.optim.Adam(model.parameters()) # use Adam
    loss_fn = nn.CrossEntropyLoss()   # use CrossEntropyLoss

    for i in range(epochs):
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """

    return model

def test_task_1_2():
    x_train_new = torch.rand(5, 784, requires_grad=True)
    y_train_new = ones = torch.ones(5, dtype=torch.uint8)
    
    assert type(train_model(x_train_new, y_train_new)) == DigitNet

# This is a demonstration: You can use this cell for exploring your trained model

idx = 0 # try on some index

scores = digit_model(x_test[idx:idx+1])
_, predictions = torch.max(scores, 1)
print("true label:", y_test[idx].item())
print("pred label:", predictions[0].item())

plt.imshow(x_test[idx].numpy().reshape(28, 28), cmap='gray')
plt.axis("off")
plt.show()

### Task 1.3 - Evaluate the model

def get_accuracy(scores: torch.Tensor, labels: torch.Tensor) -> int | float:
    """
    Helper function that returns accuracy of model
    
    Parameters
    ----------
        scores : The raw softmax scores of the network
        labels : The ground truth labels
        
    Returns
    -------
        Accuracy of the model. Return a number in range [0, 1].
        0 means 0% accuracy while 1 means 100% accuracy
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """

def test_task_1_3():
    scores = torch.tensor([[0.4118, 0.6938, 0.9693, 0.6178, 0.3304, 0.5479, 0.4440, 0.7041, 0.5573,
             0.6959],
            [0.9849, 0.2924, 0.4823, 0.6150, 0.4967, 0.4521, 0.0575, 0.0687, 0.0501,
             0.0108],
            [0.0343, 0.1212, 0.0490, 0.0310, 0.7192, 0.8067, 0.8379, 0.7694, 0.6694,
             0.7203],
            [0.2235, 0.9502, 0.4655, 0.9314, 0.6533, 0.8914, 0.8988, 0.3955, 0.3546,
             0.5752],
            [0,0,0,0,0,0,0,0,0,1]])
    y_true = torch.tensor([5, 3, 6, 4, 9])
    acc_true = 0.4
    assert isclose(get_accuracy(scores, y_true),acc_true) , "Mismatch detected"
    print("passed")

# do not remove this cell
# run this before moving on

T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

"""
Note: If you updated the path to the directory containing `MNIST` 
directory, please update it here as well.
"""
mnist_train = datasets.MNIST("./", train=True, download=False, transform=T)
mnist_test = datasets.MNIST("./", train=False, download=False, transform=T)

"""
if you feel your computer can't handle too much data, you can reduce the batch
size to 64 or 32 accordingly, but it will make training slower. 

We recommend sticking to 128 but do choose an appropriate batch size that your
computer can manage. The training phase tends to require quite a bit of memory.
"""
train_loader = torch.utils.data.DataLoader(mnist_train, shuffle=True, batch_size=256)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=10000)

# No need to edit this. Just run the cell and move on

train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()
print(f"Label: {label}")

# Demo
class RawCNN(nn.Module):
    """
    CNN model using Conv2d and MaxPool2d layers.
    """
    def __init__(self, classes: int):
        super().__init__()
        """
        classes: integer that corresponds to the number of classes for MNIST
        """
        self.conv1 = nn.Conv2d(1, 32, (3,3))
        self.mp1 = nn.MaxPool2d((2,2))
        self.lrelu = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(32, 64, (3,3))
        self.mp2 = nn.MaxPool2d((2,2))

        self.l1 = nn.Linear(64*5*5, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.lrelu(x)

        x = self.conv2(x)
        x = self.mp2(x)
        x = self.lrelu(x)   
        
        x = x.view(-1, 64*5*5) # Flattening – do not remove this line

        x = self.l1(x)
        x = self.lrelu(x)
        x = self.l2(x)
        x = self.lrelu(x)
        out = self.l3(x)
        
        return out

# Test the network's forward pass
num_samples, num_channels, width, height = 20, 1, 28, 28
x = torch.rand(num_samples, num_channels, width, height)
net = RawCNN(10)
y = net(x)
print(y.shape) # torch.Size([20, 10])

### Task 2.1: Building a ConvNet with Dropout

class DropoutCNN(nn.Module):
    """
    CNN that uses Conv2d, MaxPool2d, and Dropout layers.
    """
    def __init__(self, classes: int, drop_prob: float = 0.5):
        super().__init__()
        """
        classes: integer that corresponds to the number of classes for MNIST
        drop_prob: probability of dropping a node in the neural network
        """
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """
        
        x = x.view(-1, 64*5*5) # Flattening – do not remove

        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """

def test_task_2_1():
    # Test your network's forward pass
    num_samples, num_channels, width, height = 20, 1, 28, 28
    x = torch.rand(num_samples, num_channels, width, height)
    net = DropoutCNN(10)
    y = net(x)
    print(y.shape) # torch.Size([20, 10])

### Task 2.2: Training your Vanilla and Dropout CNNs

def train_model(loader: torch.utils.data.DataLoader, model: nn.Module):
    """
    PARAMS
    loader: the data loader used to generate training batches
    model: the model to train
  
    RETURNS
        the final trained model and losses
    """

    """
    YOUR CODE HERE
    
    - create the loss and optimizer
    """
    """ YOUR CODE HERE """
    raise NotImplementedError
    """ YOUR CODE END HERE """
    epoch_losses = []
    for i in range(10):
        epoch_loss = 0
        model.train()
        for idx, data in enumerate(loader):
            x, y = data
            """
            YOUR CODE HERE
            
            - reset the optimizer
            - perform forward pass
            - compute loss
            - perform backward pass
            """
            """ YOUR CODE HERE """
            raise NotImplementedError
            """ YOUR CODE END HERE """

        epoch_loss = epoch_loss / len(loader)
        epoch_losses.append(epoch_loss)
        print("Epoch: {}, Loss: {}".format(i, epoch_loss))
        

    return model, epoch_losses

# do not remove – nothing to code here
# run this cell before moving on
# ensure get_accuracy from task 1.5 is defined

with torch.no_grad():
    vanilla_model.eval()
    for i, data in enumerate(test_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        pred_vanilla = vanilla_model(x)
        acc = get_accuracy(pred_vanilla, y)
        print(f"vanilla acc: {acc}")
        
    do_model.eval()
    for i, data in enumerate(test_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        pred_do = do_model(x)
        acc = get_accuracy(pred_do, y)
        print(f"drop-out (0.5) acc: {acc}")
        
"""
The network with Dropout might under- or outperform the network without
Dropout. However, in terms of generalisation, we are assured that the Dropout
network will not overfit – that's the guarantee of Dropout.

A very nifty trick indeed!
"""

### Task 2.3: Observing Effects of Dropout

%%time 
# do not remove – nothing to code here
# run this before moving on

print("======Training Dropout Model with Dropout Probability 0.10======")
do10_model, do10_losses = train_model(train_loader, DropoutCNN(10, 0.10))
print("======Training Dropout Model with Dropout Probability 0.95======")
do95_model, do95_losses = train_model(train_loader, DropoutCNN(10, 0.95))

# do not remove – nothing to code here
# run this cell before moving on
# but ensure get_accuracy from task 3.5 is defined

with torch.no_grad():
    do10_model.eval()
    for i, data in enumerate(test_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        pred_do = do10_model(x)
        acc = get_accuracy(pred_do, y)
        print(acc)

    do95_model.eval()
    for i, data in enumerate(test_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        pred_do = do95_model(x)
        acc = get_accuracy(pred_do, y)
        print(acc)

densenet = nn.Sequential(
                nn.Linear(784, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
                nn.Softmax(1) # softmax dimension
            )

x = torch.rand(15, 784) # a batch of 15 MNIST images
y = densenet(x) # here we simply run the sequential densenet on the `x` tensor
print(y.shape) # a batch of 15 predictions

convnet = nn.Sequential(
                nn.Conv2d(1, 32, (3,3)),
                nn.ReLU(),
                nn.Conv2d(32, 64, (3,3)),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(36864, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
                nn.Softmax(1) # softmax dimension
            )

x = torch.rand(15, 1, 28, 28) # a batch of 15 MNIST images
y = convnet(x) # here we simply run the sequential convnet on the `x` tensor
print (y.shape) # a batch of 15 predictions

cifar_train = datasets.CIFAR10("./", train=True, download=True, transform=transforms.ToTensor())
cifar_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=128, shuffle=True)

train_features, train_labels = next(iter(cifar_train_loader))
img = train_features[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,7))
transform = transforms.Compose([transforms.RandomHorizontalFlip()
                                # YOUR CODE HERE
                                ]) # add in your own transformations to test
tensor_img = transform(img)
ax1.imshow(img.permute(1,2,0))
ax1.axis("off")
ax1.set_title("Before Transformation")
ax2.imshow(tensor_img.permute(1, 2, 0))
ax2.axis("off")
ax2.set_title("After Transformation")
plt.show()

### Task 2.4: Picking Data Augmentations

def get_augmentations() -> transforms.Compose:
    T = transforms.Compose([
        transforms.ToTensor(),
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """
    ])
    
    return T

# do not remove this cell
# run this before moving on

T = get_augmentations()

cifar_train = datasets.CIFAR10("./", train=True, download=True, transform=T)
cifar_test = datasets.CIFAR10("./", train=False, download=True, transform=T)

"""
if you feel your computer can't handle too much data, you can reduce the batch
size to 64 or 32 accordingly, but it will make training slower. 

We recommend sticking to 128 but dochoose an appropriate batch size that your
computer can manage. The training phase tends to require quite a bit of memory.

CIFAR-10 images have dimensions 3x32x32, while MNIST is 1x28x28
"""
cifar_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=128, shuffle=True)
cifar_test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=10000)

### Task 2.5: Build a ConvNet for CIFAR-10

class CIFARCNN(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        """
        classes: integer that corresponds to the number of classes for CIFAR-10
        """
        self.conv = nn.Sequential(
                        """ YOUR CODE HERE """
                        raise NotImplementedError
                        """ YOUR CODE END HERE """

                    )

        self.fc = nn.Sequential(
                        """ YOUR CODE HERE """
                        raise NotImplementedError
                        """ YOUR CODE END HERE """
                    )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """
        x = x.view(x.shape[0], 64, 6*6).mean(2) # GAP – do not remove this line
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """
        return out

%%time
# do not remove – nothing to code here
# run this cell before moving on

cifar10_model, losses = train_model(cifar_train_loader, CIFARCNN(10))

# do not remove – nothing to code here
# run this cell before moving on
# but ensure get_accuracy from task 3.5 is defined

with torch.no_grad():
    cifar10_model.eval()
    for i, data in enumerate(cifar_test_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        pred = cifar10_model(x)
        acc = get_accuracy(pred, y)
        print(f"cifar accuracy: {acc}")
        
# don't worry if the CIFAR-10 accuracy is low, it's a tough dataset to crack.
# as long as you get something shy of 50%, you should be alright!

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import matplotlib.pyplot as plt

# Set seeds for reproducibility
torch.manual_seed(2109)
np.random.seed(2109)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def rnn_cell_forward(xt, h_prev, Wxh, Whh, Why, bh, by):
    """
    Implements a single forward step of the RNN-cell

    Args:
        xt: 2D tensor of shape (nx, m)
            Input data at timestep "t"
        h_prev: 2D tensor of shape (nh, m)
            Hidden state at timestep "t-1"
        Wxh: 2D tensor of shape (nx, nh)
            Weight matrix multiplying the input
        Whh: 2D tensor of shape (nh, nh)
            Weight matrix multiplying the hidden state
        Why: 2D tensor of shape (nh, ny)
            Weight matrix relating the hidden-state to the output
        bh: 1D tensor of shape (nh, 1)
            Bias relating to next hidden-state
        by: 2D tensor of shape (ny, 1)
            Bias relating the hidden-state to the output

    Returns:
        yt_pred -- prediction at timestep "t", tensor of shape (ny, m)
        h_next -- next hidden state, of shape (nh, m)
    """
    h_next = torch.tanh(Whh.T @ h_prev + Wxh.T @ xt + bh)
    yt_pred = F.softmax(Why.T @ h_next + by, dim=0)
    return yt_pred, h_next


def generate_sine_wave(num_time_steps):
    """
    Generates a sine wave data

    Args:
        num_time_steps: int
            Number of time steps
    Returns:
        data: 1D tensor of shape (num_time_steps,)
            Sine wave data with corresponding time steps
    """
    x = torch.linspace(0, 8*torch.pi, num_time_steps)
    data = torch.sin(x)
    return data

num_time_steps = 500
sine_wave_data = generate_sine_wave(num_time_steps)

# Plot the sine wave
plt.plot(sine_wave_data)
plt.title('Sine Wave')
plt.show()

def create_sequences(sine_wave, seq_length):
    """
    Create overlapping sequences from the input time series and generate labels 
    Each label is the value immediately following the corresponding sequence.
    
    Args:
        sine_wave: A 1D tensor representing the time series data (e.g., sine wave).
        seq_length: int. The length of each sequence (window) to be used as input to the RNN.

    Returns: 
        windows: 2D tensor where each row is a sequence (window) of length `seq_length`.
        labels: 1D tensor where each element is the next value following each window.
    """
    windows = sine_wave.unfold(0, seq_length, 1)
    labels = sine_wave[seq_length:]
    return windows[:-1], labels

# Create sequences and labels
seq_length = 20
sequences, labels = create_sequences(sine_wave_data, seq_length)
# Add extra dimension to match RNN input shape [batch_size, seq_length, num_features]
sequences = sequences.unsqueeze(-1)
sequences.shape

# Split the sequences into training data (first 80%) and test data (remaining 20%) 
train_size = int(len(sequences) * 0.8)
train_seqs, train_labels = sequences[:train_size], labels[:train_size]
test_seqs, test_labels = sequences[train_size:], labels[train_size:]

### Task 3.1: Building RNN Model

class SineRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the SineRNN model.

        Args:
            input_size (int): The number of input features per time step (typically 1 for univariate time series).
            hidden_size (int): The number of units in the RNN's hidden layer.
            output_size (int): The size of the output (usually 1 for predicting a single value).
        """
        super(SineRNN, self).__init__()
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """
        
    def forward(self, x):
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """

def test_task_3_1():
    input_size = output_size = 1
    hidden_size = 50
    model = SineRNN(input_size, hidden_size, output_size).to(device)
    assert [layer.detach().numpy().shape for _, layer in model.named_parameters()]\
          == [(50, 1), (50, 50), (50,), (50,), (1, 50), (1,)]

# Define loss function, and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(train_seqs)
    loss = criterion(outputs.squeeze(), train_labels)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# Predict on unseen data
model.eval()
y_pred = []
input_seq = test_seqs[0]  # Start with the first testing sequence

with torch.no_grad():
    for _ in range(len(test_seqs)):
        output = model(input_seq)
        y_pred.append(output.item())

        # Use the predicted value as the next input sequence
        next_seq = torch.cat((input_seq[1:, :], output.unsqueeze(0)), dim=0)
        input_seq = next_seq

# Plot the true sine wave and predictions
plt.plot(sine_wave_data, c='gray', label='Actual data')
plt.scatter(np.arange(seq_length + len(train_labels)), sine_wave_data[:seq_length + len(train_labels)], marker='.', label='Train')
x_axis_pred = np.arange(len(sine_wave_data) - len(test_labels), len(sine_wave_data))
plt.scatter(x_axis_pred, y_pred, marker='.', label='Predicted')
plt.legend(loc="lower left")
plt.show()

def create_sequences_with_noise(sine_wave, sine_wave_length, noise_length):
    """
    Create overlapping sequences from the input time series and generate labels.
    Each label is the value immediately following the corresponding sequence.
    Additionally, noise of the specified length is appended to the sequences.

    Args:
        sine_wave: A 1D tensor representing the time series data (e.g., sine wave).
        sine_wave_length: int. The length of the sine wave window.
        noise_length: int. The length of noise to be appended to each sequence.

    Returns:
        windows: 2D tensor where each row is a sequence of length `sine_wave_length + noise_length`.
        labels: 1D tensor where each element is the next value following each window.
    """
    windows = sine_wave.unfold(0, sine_wave_length, 1)
    labels = sine_wave[sine_wave_length:]
    noise = torch.randn(windows.shape[0], noise_length)
    windows = torch.cat((windows, noise), dim=1)
    return windows[:-1], labels

# Create sequences and labels
sine_wave_length = 20
noise_length = 20
sequences, labels = create_sequences_with_noise(sine_wave_data, sine_wave_length, noise_length)
# Add extra dimension to match RNN input shape [batch_size, seq_length, num_features]
sequences = sequences.unsqueeze(-1)
sequences.shape

# Split the sequences into training data (first 80%) and test data (remaining 20%) 
train_size = int(len(sequences) * 0.8)
train_seqs, train_labels = sequences[:train_size], labels[:train_size]
test_seqs, test_labels = sequences[train_size:], labels[train_size:]

# Define model
input_size = output_size = 1
hidden_size = 50
model = SineRNN(input_size, hidden_size, output_size).to(device)

# Define loss function, and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(train_seqs)
    loss = criterion(outputs.squeeze(), train_labels)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

model.eval()
with torch.no_grad():
    y_pred = model(test_seqs).squeeze()
    y_true = test_labels.squeeze()

print("Test loss:", criterion(y_pred, y_true))

plt.figure(figsize=(8, 4))
plt.plot(y_true[1::2].numpy(), label="True value", color='black')
plt.plot(y_pred[1::2].numpy(), '--', label="Predicted value", color='red')
plt.title("SineRNN Predictions")
plt.xlabel("Test sequence index")
plt.ylabel("Target value")
plt.legend()
plt.show()

## Task 4.1: Positional Encoding Layer

class PositionalEncoding(nn.Module):
    def __init__(self):
        # You do not need to change anything in this function.
        super(PositionalEncoding, self).__init__()

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.

        You should use vectorized operations to compute the positional encoding.
        The use of Python loops is not allowed.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size)
        """
        """ YOUR CODE HERE """
        raise NotImplementedError
        """ YOUR CODE END HERE """

def test_task_4_1():
    encoder = PositionalEncoding()
    x0 = torch.zeros((1, 2, 4))
    y0 = encoder(x0)
    a0 = torch.tensor([[[0.0000, 1.0000, 0.0000, 1.0000],
                        [0.8415, 0.5403, 0.0100, 0.9999]]])
    assert torch.allclose(y0, a0, atol=1e-4)
    
    x1 = torch.ones((1, 4, 6))
    y1 = encoder(x1)
    a1 = torch.tensor([[[1.0000, 2.0000, 1.0000, 2.0000, 1.0000, 2.0000],
                        [1.8415, 1.5403, 1.0464, 1.9989, 1.0022, 2.0000],
                        [1.9093, 0.5839, 1.0927, 1.9957, 1.0043, 2.0000],
                        [1.1411, 0.0100, 1.1388, 1.9903, 1.0065, 2.0000]]])
    assert torch.allclose(y1, a1, atol=1e-4)

class TransformerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the TransformerNN model. We use the same hidden size for the feedforward network and the Transformer encoder.

        Args:
            input_size (int): The number of input features per time step (typically 1 for univariate time series).
            hidden_size (int): The number of units in the Transformer's hidden layers.
            output_size (int): The size of the output (usually 1 for predicting a single value).
        """
        super(TransformerNN, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.positional_encoder = PositionalEncoding()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, dim_feedforward=hidden_size, nhead=1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoder(x)
        x = self.transformer_encoder(x)

        # The encoder outputs a sequence of hidden states, so
        # we take the mean across the sequence length dimension.
        x = x.mean(dim=1)

        out = self.fc_out(x)
        return out

model = TransformerNN(input_size=1, hidden_size=50, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_seqs)
    loss = criterion(outputs.squeeze(), train_labels)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')


model.eval()
with torch.no_grad():
    y_pred = model(test_seqs).squeeze()
    y_true = test_labels.squeeze()

print("Test loss:", criterion(y_pred, y_true))

plt.figure(figsize=(8, 4))
plt.plot(y_true[1::2].numpy(), label="True value", color='black')
plt.plot(y_pred[1::2].numpy(), '--', label="Predicted value", color='red')
plt.title("TransformerNN Predictions")
plt.xlabel("Test sequence index")
plt.ylabel("Target value")
plt.legend()
plt.show()

## Task 4.2: Visualizing Attention Scores

## Task 4.3: Comparing Transformers with or without Positional Encoding

# You may use this cell and create new cells to experiment.


if __name__ == '__main__':
    test_task_1_1()
    test_task_1_2()
    test_task_1_3()
    test_task_2_1()
    test_task_3_1()
    test_task_4_1()