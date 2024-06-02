import numpy as np
class Dense():
    def __init__(self, no_of_neurons, input_size=1, activation='linear', gain=1):
        self.no_of_neurons=no_of_neurons
        self.input_size=input_size
        self.activation=activation
        self.gain=gain

        self.weights=np.random.randn(input_size,no_of_neurons) * (gain/(input_size)**0.5)
        self.biases=np.zeros((1,no_of_neurons))

        self.gradient_w=np.zeros((input_size,no_of_neurons))
        self.gradient_b=np.zeros((1,no_of_neurons))

    def forward(self,inputs):
        z = np.dot(inputs,self.weights)+self.biases
        match self.activation:
            case "linear": return z
            case "sigmoid": return self.sigmoid(z)
            case "relu": return self.relu(z)
            case "tanh": return self.tanh(z)
            case "softmax": return self.softmax(z)

    def gradient_descent(self,lr):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j]-=lr*self.gradient_w[i][j]

        self.biases-=lr*self.gradient_b

    def sigmoid(self,inputs):
        activations = np.exp(np.fmin(inputs, 0)) / (1 + np.exp(-np.abs(inputs)))
        return activations

    def relu(self,inputs):
        activations = np.maximum(np.zeros(inputs.shape),inputs)
        return activations

    def tanh(self,inputs):
        activations = np.tanh(inputs)
        return activations

    def softmax(self, inputs):
        expo = np.exp(inputs -  np.max(inputs, axis=1, keepdims=True))
        activations = expo / np.sum(expo, axis=1, keepdims=True)
        return activations


class Network():
    def __init__(self, layers):
        self.layers=[]
        self.costs=[]
        self.layers.append(Dense(layers[0].no_of_neurons, input_size=layers[0].input_size, activation=layers[0].activation, gain=layers[0].gain))
        for i in range(1,len(layers)):
            self.layers.append(Dense(layers[i].no_of_neurons,input_size=layers[i-1].no_of_neurons, activation=layers[i].activation, gain=layers[i].gain))

    def forward(self,inputs):
        for layer in self.layers:
            inputs=layer.forward(inputs)
        return inputs

    def compile(self, loss, lr):
        self.loss=loss
        self.lr=lr

    def fit(self, x, y, epochs):
        for _ in range(epochs):
            self.calculate_gradient_of_cost(x,y)
            for layer in self.layers:
                layer.gradient_descent(self.lr)
            self.costs.append(self.cost(x,y))

    def cost(self,samples,labels):
        outputs = self.forward(samples)
        if self.loss == 'mse':
            return np.mean(np.sum(((outputs-labels)**2),axis=1,keepdims=True))

        elif self.loss == 'categorical_crossentropy':
            clipped_outputs = np.clip(outputs, 1e-15, 1 - 1e-15)
            correct_class_probs = clipped_outputs[np.arange(samples.shape[0]), labels]
            return np.mean(-np.log(correct_class_probs))

        else:
            print("Specify a valid loss function(mse or categorical_crossentropy)")

    def calculate_gradient_of_cost(self,inputs,labels):
        h=0.0001
        original_cost=self.cost(inputs,labels)
        for layer in self.layers:
            for i in range(layer.no_of_neurons):
                for j in range(layer.input_size):
                    layer.weights[j][i]+=h
                    layer.gradient_w[j][i]=(self.cost(inputs,labels)-original_cost)/h
                    layer.weights[j][i]-=h
                layer.biases[0][i]+=h
                layer.gradient_b[0][i]=(self.cost(inputs,labels)-original_cost)/h
                layer.biases[0][i]-=h