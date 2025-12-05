import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

x = np.array([[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(42)

input_layer_neuron = x.shape[1]
hidden_layer_neurons = 2
output_neurons = 1

wh = np.random.uniform(size=(input_layer_neuron, hidden_layer_neurons))
bh = np.random.uniform(size=(1, hidden_layer_neurons))
wo = np.random.uniform(size=(hidden_layer_neurons, output_neurons))
bo = np.random.uniform(size=(1, output_neurons))

epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    # Forward Propagation
    hidden_layer_activation = np.dot(x, wh) + bh
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, wo) + bo
    predicted_output = sigmoid(output_layer_activation)

    # Backward Propagation
    error = y - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(wo.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating Weights and Biases
    wo += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bo += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    wh += x.T.dot(d_hidden_layer) * learning_rate
    bh += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

print("Predicted Output:")
print(predicted_output)

