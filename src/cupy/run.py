from network import Network
import mnist_loader
import matplotlib.cm as cm
import matplotlib.pyplot as plt


training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = Network([784, 16, 10])
print("Created network")
net.SGD(training_data, 30, 10, 1.0, test_data=test_data)

raw_pixels, pixel_vector, expected = mnist_loader.load_random_image()


def toNumber(array):
    array = array.reshape(10)
    max = 0
    for pos in range(1, 10):
        if (array[pos] > array[max]):
            max = pos
    return max


result = net.feedforward(pixel_vector)
print(toNumber(result))
print("Is correct? {}".format(expected == toNumber(result)))


plt.imshow(raw_pixels.reshape((28, 28)), cmap=cm.Greys_r)
plt.show()
