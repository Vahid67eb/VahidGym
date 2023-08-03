'''Certainly! Here's an example of how to create a simple neural network using PyTorch:'''

```python
import torch
import torch.nn as nn

# Define a neural network class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Create an instance of the neural network
input_size = 10
hidden_size = 20
output_size = 5
model = NeuralNetwork(input_size, hidden_size, output_size)

# Print the model architecture
print(model)
```

''' In this example, we first import the necessary modules: `torch` for tensor operations and `torch.nn` for neural network-related functionality.

Next, we define a custom neural network class called `NeuralNetwork` that inherits from `nn.Module`, the base class for all neural network modules in PyTorch. Inside the class, we define the network architecture in the `__init__` method. In this case, the network consists of two fully connected layers with ReLU activation in between.

The `forward` method defines the forward pass of the network. It specifies how the input tensor `x` flows through the network layers to produce the output.

After defining the neural network class, we create an instance of the `NeuralNetwork` class, specifying the input size, hidden size, and output size.

Finally, we print the model architecture, which outputs the structure of the neural network: '''

```
NeuralNetwork(
  (fc1): Linear(in_features=10, out_features=20, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=20, out_features=5, bias=True)
)
```

'''This shows that the neural network consists of two linear (fully connected) layers, with input size 10, hidden size 20, and output size 5, along with ReLU activation between the layers.
