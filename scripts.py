import numpy as np

def predict(X, w):
    return X * w

def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)

def train(X, Y, iterations, lr):
    w = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w)
        print("Iteration %4d => Loss: %.6f" % (i, current_loss))
        if loss(X, Y, w + lr) < current_loss:
            w += lr
        elif loss(X, Y, w - lr) < current_loss:
            w -= lr
        else:
            return w

    raise Exception("Couldn't converge within %d iterations" % iterations)

def predict_bias(X, w, b):
    return X * w + b

def loss_bias(X, Y, w, b):
    return np.average((predict_bias(X, w, b) - Y) ** 2)

def train_bias(X, Y, iterations, lr):
    w = b = 0
    for i in range(iterations):
        current_loss = loss_bias(X, Y, w, b)
        if i % 300 == 0:
            print("Iteration %4d => Loss: %.6f" % (i, current_loss))

        if loss_bias(X, Y, w + lr, b) < current_loss:
            w += lr
        elif loss_bias(X, Y, w - lr, b) < current_loss:
            w -= lr
        elif loss_bias(X, Y, w, b + lr) < current_loss:
            b += lr
        elif loss_bias(X, Y, w, b - lr) < current_loss:
            b -= lr
        else:
            return w, b

    raise Exception("Couldn't converge within %d iterations" % iterations)