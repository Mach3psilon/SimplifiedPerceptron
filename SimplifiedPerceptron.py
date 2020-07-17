"""




We want to learn such functions:
    f(x) = w * x + b
where w and b are constant values (e.g. f(x) = 5 * x - 7).

For this, we will define a very simple perceptron (single input, single output).

Homework:

Generalize this simple perceptron to a regular perceptron (i.e. multiple inputs).

Our functions can be:
    f(x1, x2, ..., xN) = w1 * x1 + w2 * x2 + ... + wN * xN + b
where w1, w2, ..., wN and b are constant values (e.g. f(x1, x2) = 3 * x1 - 2 * x2 + 10).

"""

import random

# training set
train_X = [1, 5, 10, 30, -3, 0, 40, 50, 60, -10] # , 100, -100]
train_Y = [-2, 18, 43, 143, -22, -7, 193, 293, -57] # , 493, -507]

# test set
test_X = [2, 7, 19, 21, 12, -4, 11, 0]
test_Y = [3, 28, 88, 98, 53, -27, 48, -7]

class Perceptron(object):

    def __init__(self):
        self.w = random.uniform(-1, 1)
        self.b = random.uniform(-1, 1)

    def forward(self, x):
        p = self.w * x + self.b
        return p

    def backward(self, x, p, y, lr):
        dc_dp = self.loss(p, y, derivative=True)
        dp_dw = x
        dp_db = 1

        dc_dw = dc_dp * dp_dw
        dc_db = dc_dp * dp_db

        self.w -= lr * dc_dw
        self.b -= lr * dc_db

    def loss(self, p, y, derivative=False):
        if derivative:
            # with respect to p
            if p > y:
                return 1
            else:
                return -1 # p=y ...
        else:
            return abs(p - y)

    def train(self, X, Y, lr=0.001, epoch=1):
        avg_error_by_epoch = []
        for i in range(epoch):

            # Shuffle X and Y together.
            # Because the order of data matters for training.
            #combined = list(zip(X, Y))
            #random.shuffle(combined)
            #X[:], Y[:] = zip(*combined)

            avg_error = 0
            for x, y in zip(X, Y):
                p = self.forward(x)
                error = self.loss(p, y)
                self.backward(x, p, y, lr)
                avg_error += error

            avg_error = avg_error / len(X)
            avg_error_by_epoch.append(avg_error)

        return avg_error_by_epoch  # just for monitoring

    def test(self, X, Y):
        avg_error = 0
        predictions = []
        for x, y in zip(X, Y):
            p = self.forward(x)
            predictions.append(p)
            error = self.loss(p, y)
            avg_error += error
        avg_error = avg_error / len(X)
        return predictions, avg_error


learning_rate = 0.0001
num_epoch = 20000
random.seed(0)

P = Perceptron()
avg_error_by_epoch = P.train(train_X, train_Y, lr=learning_rate, epoch=num_epoch)

print("Training error:", avg_error_by_epoch)

test_predictions, avg_test_error = P.test(test_X, test_Y)



# Random restarts:
best_perceptron = None
best_avg_error = float('inf')
for _ in range(10):
    P = Perceptron()
    avg_error_by_epoch = P.train(train_X, train_Y, lr=learning_rate, epoch=num_epoch)
    print(avg_error_by_epoch[-1])
    if avg_error_by_epoch[-1] < best_avg_error:
        best_avg_error = avg_error_by_epoch[-1]
        best_perceptron = P
        
test_predictions, avg_test_error = best_perceptron.test(test_X, test_Y)


print("Targets:", test_Y)
print("Predictions:", test_predictions)
print("Test error:", avg_test_error)

# if you don't have matplotlib, you can remove the related lines below.

import matplotlib.pyplot as plt

step = 10 # Do not show every loss value on the graph (because there are many of them), instead show 1 point for every 100 points.
x_axis = range(1, num_epoch + 1, step)  # x_axis = range(1, num_epoch + 1)
plt.scatter(x_axis, avg_error_by_epoch[::step])  # plt.scatter(x_axis, avg_error_by_epoch)
plt.ylim(ymin=0)
plt.show()

a = input("Stop target")
print(a)
# A local minima is found.
