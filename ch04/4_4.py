import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # ガウス分布で初期化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        retrun loss


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # xと同じ形状の配列を生成

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)の計算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)の計算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad

# def f(W):
#     return net.loss(x, t)

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

def function_2(x):
    return x[0]**2 + x[1]**2

# init_x = np.array([-3.0, 4.0])
# gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)

net = simpleNet()
print(net.W) # 重みパラメータ

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
np.argmax(p) # 最大値のインデックス

t = np.array([0, 0, 1]) # 正解ラベル
net.loss(x, t)

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)
print(dW)
