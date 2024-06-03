# Deadrin
![Logo](images/logo.png)<br>
Deadrin is a simple neural network library that computes gradients without using backpropagation. It uses a brute-force method to estimate gradients by making tiny change to each weight and bias and observing the effect on the cost function. While this method works, it is highly inefficient for large networks and datasets.
### The process is as follows:
- For each weight and bias in the network, `Deadrin` makes a tiny change to that parameter.
- It then evaluates the cost function (which measures how well the network is performing) before and after the tiny change.
- The gradient is estimated by observing how the cost function changes due to this tiny change.
- This process is repeated for every weight and bias in the network.
- Once all gradients are computed, `Deadrin` uses gradient descent to update all the weights and biases, moving them in the direction that reduces the cost function.

  
## Features
- Simple Dense layer implementation with various activation functions
- Brute-force gradient computation
- Supports mean squared error (MSE) and categorical cross-entropy loss functions
- Educational tool to understand basic neural network training concepts


## Dependencies
- [numpy](https://numpy.org/install/)

## Installation
```shell
pip install deadrin
```

## Demo
 ```python
import numpy as np
from deadrin import Dense, Network

# Create the network
model = Network([
      Dense(no_of_neurons=3, input_size=2, activation='relu')
      Dense(no_of_neurons=1, activation='sigmoid')
])

# Compile the network
model.compile(loss='mse', lr=0.01)

# Generate some dummy data
X_train = np.random.randn(100, 2)
y_train = np.random.randint(0, 2, (100, 1))

# Train the network
model.fit(X_train, y_train, epochs=1000)

# Make predictions
predictions = model.forward(X_train)
print(predictions)
```

### Contributing
Contributions are most welcome! Please open an issue or submit a pull request.

### License
Deadrin is released under the MIT License.
