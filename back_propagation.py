import numpy as np
import matplotlib.pyplot as plt


# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# XOR 
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])  

np.random.seed(42)
input_neurons = 2
hidden1_neurons = 4
hidden2_neurons = 4
output_neurons = 1

# He initialization
weights_input_hidden1 = np.random.randn(input_neurons, hidden1_neurons) * np.sqrt(2.0 / input_neurons)
weights_hidden1_hidden2 = np.random.randn(hidden1_neurons, hidden2_neurons) * np.sqrt(2.0 / hidden1_neurons)
weights_hidden2_output = np.random.randn(hidden2_neurons, output_neurons) * np.sqrt(2.0 / hidden2_neurons)

# Initialise biases
bias_hidden1 = np.zeros((1, hidden1_neurons))
bias_hidden2 = np.zeros((1, hidden2_neurons))
bias_output = np.zeros((1, output_neurons))

# Training parameters
epochs = 10000
learning_rate = 0.1
loss_history = []  

print("Starting XOR training...\n")

for epoch in range(epochs):
    # Forward propagation
    hidden1_input = np.dot(X, weights_input_hidden1) + bias_hidden1
    hidden1_output = sigmoid(hidden1_input)
    
    hidden2_input = np.dot(hidden1_output, weights_hidden1_hidden2) + bias_hidden2
    hidden2_output = sigmoid(hidden2_input)
    
    output_input = np.dot(hidden2_output, weights_hidden2_output) + bias_output
    predicted_output = sigmoid(output_input)
    
    # Calculate error
    error = y - predicted_output
    loss = np.mean(np.square(error))  
    loss_history.append(loss)
    
    # Backward propagation
    d_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden2 = d_output.dot(weights_hidden2_output.T)
    d_hidden2 = error_hidden2 * sigmoid_derivative(hidden2_output)
    
    error_hidden1 = d_hidden2.dot(weights_hidden1_hidden2.T)
    d_hidden1 = error_hidden1 * sigmoid_derivative(hidden1_output)
    
    # Update weights and biases
    weights_hidden2_output += hidden2_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    
    weights_hidden1_hidden2 += hidden1_output.T.dot(d_hidden2) * learning_rate
    bias_hidden2 += np.sum(d_hidden2, axis=0, keepdims=True) * learning_rate
    
    weights_input_hidden1 += X.T.dot(d_hidden1) * learning_rate
    bias_hidden1 += np.sum(d_hidden1, axis=0, keepdims=True) * learning_rate
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

print("\n" + "="*50)
print("XOR Training Results:")
print("="*50)

# Testing
print("\nTest predictions:")
for i in range(len(X)):
    actual = y[i][0]
    predicted = predicted_output[i][0]
    predicted_class = 1 if predicted > 0.5 else 0
    print(f"Input: {X[i]} -> Expected: {actual}, Predicted: {predicted:.4f} (class {predicted_class})")

print(f"\nClassification threshold: 0.5")

# Visual
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.subplot(1, 2, 2)
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
grid = np.c_[xx.ravel(), yy.ravel()]

hidden1 = sigmoid(np.dot(grid, weights_input_hidden1) + bias_hidden1)
hidden2 = sigmoid(np.dot(hidden1, weights_hidden1_hidden2) + bias_hidden2)
Z = sigmoid(np.dot(hidden2, weights_hidden2_output) + bias_output)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.7)
plt.colorbar(label='Class 1 Probability')
plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='RdBu', edgecolors='black', s=100)
plt.title("Neural Network Decision Boundary")
plt.xlabel("X1")
plt.ylabel("X2")
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)

plt.tight_layout()
plt.show()

