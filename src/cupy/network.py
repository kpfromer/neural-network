# Standard library
import random

# Third-party libraries
import cupy as cp


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        #
        self.weights = [cp.random.randn(y, x)
                        for x, y in zip(sizes[0:-1], sizes[1:])]
        # create biases (x by 1) for layer 1 to last layer
        self.biases = [cp.random.randn(x, 1) for x in sizes[1:]]

    # a = input vector
    def feedforward(self, a):
        # for every layer
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(cp.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""

        # instead of list use numpy for top level weights

        mini_batch_size = len(mini_batch)

        # (number of images, input layer activation values)
        xs = cp.array([x for x, y in mini_batch]).transpose().reshape(
            self.sizes[0], mini_batch_size)
        # (number of images, expected output layer values)
        ys = cp.array([y for x, y in mini_batch]).transpose().reshape(
            self.sizes[-1], mini_batch_size)

        nabla_weight, nabla_bias = self.backprop(xs, ys, mini_batch_size)

        # nabla_bias was a matrix with the biases as rows and mini_batch_size
        # number of columns. We must flatten them
        for layer in range(0, len(nabla_bias)):
            # sum along the rows
            biases = nabla_bias[layer].sum(axis=1)
            bias_count = biases.shape[0]
            # restructure back to node count x 1
            nabla_bias[layer] = biases.reshape((bias_count, 1))

        # move the in opposite (down the hill) of the gradient of the cost
        self.weights = [w - (eta / len(mini_batch)) * dnw for dnw,
                        w in zip(nabla_weight, self.weights)]
        self.biases = [b - (eta / len(mini_batch)) * dnb for dnb,
                       b in zip(nabla_bias, self.biases)]

    def backprop(self, xs, ys, mini_batch_size):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        # feed foward
        activation = xs
        activations = [xs]
        zs = []

        for w, b in zip(self.weights, self.biases):
            # bs = [b, b, b, ... len(mini_batch)] create column of biases for
            # every image in mini_batch
            bs = cp.tile(b, (1, mini_batch_size))
            z = cp.dot(w, activation) + bs
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # calculate error for last layer
        nabla_bias = [cp.zeros(b.shape) for b in self.biases]
        nabla_weight = [cp.zeros(w.shape) for w in self.weights]

        delta = self.cost_derivative(
            activations[-1], ys) * sigmoid_prime(zs[-1])
        nabla_bias[-1] = delta
        nabla_weight[-1] = cp.dot(delta, activations[-2].transpose())

        # back propgate the error
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = cp.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_bias[-l] = delta
            # nabla_weight[-l] = activations[-l] * delta
            # print delta.shape
            # print activations[-l - 1].shape
            # print "e"
            nabla_weight[-l] = cp.dot(delta, activations[-l - 1].transpose())
            # nabla_w[-l] = cp.dot(delta, activations[-l - 1].transpose())

        return (nabla_weight, nabla_bias)

    def evaluate(self, test_data):
        test_results = [(cp.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y


# Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + cp.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
