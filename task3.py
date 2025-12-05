#--1

import numpy as np
from matplotlib import pyplot as plt

def sig(x):
    return 1/(1+np.exp(-x))

def dsig(x):
    return sig(x) * (1- sig(x))

x_data = np.linspace(-6,6,100)
y_data = sig(x_data)
dy_data = dsig(x_data)

plt.plot(x_data, y_data, x_data, dy_data)
plt.title('Sigmoid Function & Derivative')
plt.legend(['f(x)','f\'(x)'])
plt.grid()
plt.show()


#--2

import numpy as np
from matplotlib import pyplot as plt

def relu(x):
    temp = [max(0,value) for value in x]
    return np.array(temp, dtype=float)

def drelu(x):
    temp = [1 if value>0 else 0 for value in x]
    return np.array(temp, dtype=float)

x_data = np.linspace(-6,6,100)
y_data = relu(x_data)
dy_data = drelu(x_data)

plt.plot(x_data, y_data, x_data, dy_data)
plt.title('RELU & Derivative')
plt.legend(['f(x)','f\'(x)'])
plt.grid()
plt.show()


#--3

import numpy as np
from matplotlib import pyplot as plt

def hyp(x):
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def dhyp(x):
    return 1 - hyp(x) * hyp(x)

x_data = np.linspace(-6,6,100)
y_data = hyp(x_data)
dy_data = dhyp(x_data)

plt.plot(x_data, y_data, x_data, dy_data)
plt.title('Hyperbolic Tangent & Derivative')
plt.legend(['f(x)','f\'(x)'])
plt.grid()
plt.show()