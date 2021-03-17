import numpy as np
import math

def print_hello_world():
    test = "Hello world !"
    print("test: " + test)

def sigmoid_math(x):
    s = 1 / (1 + math.exp(-x))
    return s

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def sigmoid_deriv(x):
    s = sigmoid(x)
    ds = s * (1 - s)
    return ds


def image_to_vector(image):
    v = image.reshape((image.shape[0]*image.shape[1], image.shape[2]))
    return v

def normalize_rows(x):
    x_norm = np.linalg.norm(x, axis = 1, keepdims = True)
    x = x / x_norm
    return x

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp / x_sum
    print(x_exp.shape, x_sum.shape, s.shape)
    return s


def L1_function(y_predicted, y):
    L1 = np.sum(np.abs(y_predicted - y),axis = 0)
    return L1

def L2_function(y_predicted, y):
    L2 = np.dot(np.abs(y_predicted - y), np.abs(y_predicted -y))
    return L2

def main():
    # print(sigmoid_math(2.5))
    # x = np.array([1.5, 2.5, 3.5])
    # print(sigmoid(x))
    # x = np.array([1.5, 2.5, 3.5])
    # print ("sigmoid_deriv(x) = " + str(sigmoid_deriv(x)))
    # image = np.array([[[ 0.67126139,  0.29381281],
    #     [ 0.90714982,  0.52835547],
    #     [ 0.42245251 ,  0.45012151]],

    #    [[ 0.92814219,  0.96677647],
    #     [ 0.85114703,  0.52351845],
    #     [ 0.19981397,  0.27417313]],

    #    [[ 0.6213595,  0.00531265],
    #     [ 0.1210313,  0.49974237],
    #     [ 0.3432129,  0.94631277]]])
    # print ("image_to_vector(image) = " + str(image_to_vector(image)))
    # x = np.array([
    # [2, 3, 6],
    # [5, 2, 8]])
    # print("normalize_rows(x) = " + str(normalize_rows(x)))
    # x_vect = np.array([[9, 4, 0, 0 ,0]])
    # print("softmax(x_vect) = " + str(softmax(x_vect)))
    # x_matr = np.array([
    #     [1, 7, 5, 0, 6],
    #     [3, 4, 0, 2 ,0]])
    # print("softmax(x_matr) = " + str(softmax(x_matr)))
    # y_predicted = np.array([.8, 0.3, 0.2, .6, .2])
    # y = np.array([1, 1, 0, 1, 0])
    # print("L1 = " + str(L1_function(y_predicted,y)))
    y_predicted = np.array([.8, 0.3, 0.2, .6, .2])
    y = np.array([1, 1, 0, 1, 0])
    print("L2 = " + str(L2_function(y_predicted,y)))
main()