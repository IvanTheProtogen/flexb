import math

class neuron:
    def __init__(self, func, deriv, weights, bias):
        self.func = func
        self.deriv = deriv
        self.weights = weights
        self.bias = bias
        self.output = None
        self.input = None
        self.error = None

    def activate(self, input):
        self.input = input
        abc = sum(x * w for x, w in zip(input, self.weights)) + self.bias
        self.output = self.func(abc)
        return self.output

class network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input):
        for layer in self.layers:
            output = []
            for neur in layer:
                output.append(neur.activate(input))
            input = output
        return input

    def backward(self, target, learnRate):
        # sowwy i used ai for this :<
        output_layer = self.layers[-1]
        for i in range(len(output_layer)):
            neuron = output_layer[i]
            z = sum(w * x for w, x in zip(neuron.weights, neuron.input)) + neuron.bias
            delta = (neuron.output - target[i]) * neuron.deriv(z)
            neuron.error = delta
        for layer_idx in reversed(range(len(self.layers) - 1)):
            current_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]
            for i, neuron in enumerate(current_layer):
                error_sum = 0.0
                for next_neuron in next_layer:
                    error_sum += next_neuron.error * next_neuron.weights[i]
                z = sum(w * x for w, x in zip(neuron.weights, neuron.input)) + neuron.bias
                delta = error_sum * neuron.deriv(z)
                neuron.error = delta
        for layer in self.layers:
            for neuron in layer:
                for j in range(len(neuron.weights)):
                    neuron.weights[j] -= learnRate * neuron.error * neuron.input[j]
                neuron.bias -= learnRate * neuron.error

class funcs:
    def blank(self,x):
        return x
    def blankDeriv(self,x):
        return 1
    def relu(self,x):
        return max(0, x)
    def reluDeriv(self,x):
        return 1 if x > 0 else 0
    def sigmoid(self,x):
        return 1 / (1 + math.exp(-x))
    def sigmoidDeriv(self,x):
        s = funcs.sigmoid(x)
        return s * (1 - s)
    def tanh(self,x):
        return math.tanh(x)
    def tanhDeriv(self,x):
        return 1 - math.tanh(x) ** 2
    def swish(self,x):
        return x * funcs.sigmoid(x)
    def swishDeriv(self,x):
        s = funcs.sigmoid(x)
        return s + x * s * (1 - s)

funcs = funcs()
