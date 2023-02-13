import numpy as np
from matplotlib import pyplot as plt
from keras.datasets import mnist


def init_params():
    w1 = np.random.rand(10, 784) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    w3 = np.random.rand(10, 10) - 0.5

    b1 = np.random.rand(10, 1) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    b3 = np.random.rand(10, 1) - 0.5

    return w1, w2, w3, b1, b2, b3

def one_hot_encode(x):
    one_hot = np.zeros((x.max() + 1, x.size))
    one_hot[x, np.arange(x.size)] = 1
    
    return one_hot

def ReLU(x):
    return np.maximum(x, 0)

def ReLU_prime(x):
    return x > 0

def softmax(x):
    exp = np.exp(x )
    return exp / exp.sum(axis=0)

def feedforward(x, w1, w2, w3, b1, b2, b3):
    input = x

    z1 = np.dot(w1, input) + b1
    layer1 = ReLU(z1)

    z2 = np.dot(w2, layer1) + b2
    layer2 = ReLU(z2)

    z3 = np.dot(w3, layer2) + b3
    output = softmax(z3)

    return z1, z2, z3, layer1, layer2, output

def cost_prime(a, y):
    return 2 * (a - y)

def backpropagation(input, y, layer1, layer2, output, w1, w2, w3, z1, z2, z3, m):
    y_one_hot = one_hot_encode(y)

    output_error = cost_prime(output, y_one_hot)
    layer2_error = np.dot(w3.T, output_error) * ReLU_prime(z2)
    layer1_error = np.dot(w2.T, layer2_error) * ReLU_prime(z1)

    dW1 = 1/m * np.dot(layer1_error, input.T)
    dW2 = 1/m * np.dot(layer2_error, layer1.T)
    dW3 = 1/m * np.dot(output_error, layer2.T)

    dB1 = 1/m * np.sum(layer1_error, 1)
    dB2 = 1/m * np.sum(layer2_error, 1)
    dB3 = 1/m * np.sum(output_error, 1)

    return dW1, dW2, dW3, dB1, dB2, dB3

def update(w1, w2, w3, b1, b2, b3, dW1, dW2, dW3, dB1, dB2, dB3, alpha):
    w1 = w1 - alpha * dW1
    w2 = w2 - alpha * dW2
    w3 = w3 - alpha * dW3

    b1 = b1 - alpha * np.reshape(dB1, (10, 1))
    b2 = b2 - alpha * np.reshape(dB2, (10, 1))
    b3 = b3 - alpha * np.reshape(dB3, (10, 1))

    return w1, w2, w3, b1, b2, b3

def accuracy(pred, y):
    return np.sum(pred == y) / y.size

def gradient_descent(x_train, y_train, alpha, epochs):
    ## Train Model
    _, m = x_train.shape

    w1, w2, w3, b1, b2, b3 = init_params()

    for i in range(epochs + 1):
        z1, z2, z3, layer1, layer2, output = feedforward(x_train, w1, w2, w3, b1, b2, b3)
        dW1, dW2, dW3, dB1, dB2, dB3 = backpropagation(x_train, y_train, layer1, layer2, output, w1, w2, w3, z1, z2, z3, m)
        w1, w2, w3, b1, b2, b3 = update(w1, w2, w3, b1, b2, b3, dW1, dW2, dW3, dB1, dB2, dB3, alpha)
        if i % 10 == 0:
            print("Iteration:", i)
            predictions = np.argmax(output, 0)
            print("\tPredictions:", predictions)
            print("\tLabels:", y_train)
            print("\tAccuracy:", accuracy(predictions, y_train))

    return w1, w2, w3, b1, b2, b3

## Testing
def make_predictions(x, w1, w2, w3 ,b1, b2, b3):
    _, _, _, _, _, output = feedforward(x, w1, w2, w3, b1, b2, b3)
    predictions = np.argmax(output, 0)
    return predictions

def test_predictions(x_test, y_test, w1, w2, w3, b1, b2, b3):
    predictions = make_predictions(x_test, w1, w2, w3, b1, b2, b3)
    print("\tAccuracy:", accuracy(predictions, y_test))

    # current_image = image.reshape((WIDTH, HEIGHT)) / 255

    # plt.gray()
    # plt.imshow(current_image, interpolation='nearest')
    # plt.show()

def test_single_prediction(index, x, y, w1, w2, w3, b1, b2, b3):
    current_image = x[:, index, None]
    prediction = make_predictions(x[:, index, None], w1, w2, w3, b1, b2, b3)
    label = y[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((WIDTH, HEIGHT)) / 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


(x_train, y_train), (x_test, y_test) = mnist.load_data() # get MNIST data and split into training and test sets
WIDTH = x_train.shape[1]
HEIGHT = x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], WIDTH * HEIGHT).T / 255 # flatten training inputs into 6000 784x1 column vectors
print(x_test.shape)
WIDTH = x_test.shape[1]
HEIGHT = x_test.shape[2]
x_test = x_test.reshape(x_test.shape[0], WIDTH * HEIGHT).T / 255 #flatten testing inputs into 600 784x1 column vectors

print("--------TRAINING---------------------------------------------------------------")
w1, w2, w3, b1, b2, b3 = gradient_descent(x_train, y_train, 0.15, 500)
print("--------TESTING----------------------------------------------------------------")
test_predictions(x_test, y_test, w1, w2, w3, b1, b2, b3)

test_single_prediction(7154, x_test, y_test, w1, w2, w3, b1, b2, b3)