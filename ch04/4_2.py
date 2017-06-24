import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# def cross_entropy_error(y, t):
#     delta = 1e-7
#     return -np.sum(t * np.log(y + delta))

# def cross_entropy_error(y, t):
#     if y.dim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
#
#     batch_size = y.shape[0]
#     return -np.sum(t * np.log(y)) / batch_size

def cross_entropy_errori(y, t:
    if y.dim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size

(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000, 10)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
