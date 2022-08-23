# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff

import numpy as np
import pandas as pd
import math


def euclidean_distance(x1, x2):
    """
    Computes and returns the Euclidean distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """ 
    
    arr_sum = np.add(x1,x2)
    
    arr_sqared = np.square(arr_sum)
    
    dist = math.sqrt(np.sum(arr_sqared))
    
    return dist
    #raise NotImplementedError('This function must be implemented by the student.')


def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    
    arr_diff = np.subtract(x2, x1)
    
    abs_of_diff = np.abs(arr_diff)
    
    dist = np.sum(abs_of_diff)
    
    return dist
    #raise NotImplementedError('This function must be implemented by the student.')


def identity(x, derivative = False):
    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    if derivative == True:
        return np.ones(x.shape, dtype=float)
    
    return x
    #raise NotImplementedError('This function must be implemented by the student.')


def sigmoid(x, derivative = False):
    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    
    if derivative == True:
        
        sig = tanh(x * 0.5, derivative=True) * 0.5 + 0.5
    else:
 
        sig = tanh(x * 0.5) * 0.5 + 0.5
    return sig 
    
    """ 
    if derivative == True:
        sig = tanh(x * 0.5, derivative=True) * 0.5 + 0.5
        return sig
    #sig = tanh(x * 0.5) * 0.5 + 0.5
    return (1.0+np.tanh(x/2.0))/2.0
    """
    
    

    #raise NotImplementedError('This function must be implemented by the student.')


def tanh(x, derivative = False):
    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    
    #raise NotImplementedError('This function must be implemented by the student.')
    """
    result = np.tanh(x)
    if derivative == True:
        return 1 - (result*result)
    return result
    """
    if derivative == True:
        return 1-np.tanh(x)**2
    return np.tanh(x)


def relu(x, derivative = False):
    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.
    
    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    """
    expr = np.maximum(x, 0)
    if derivative == True:
        return np.gradient(expr, x)
    return expr
    """
    
    
    if (derivative == True):# the derivative of the ReLU is the Heaviside Theta
        expr = np.heaviside(x, 1)
    else :
        expr = np.maximum(x, 0)
   
    #raise NotImplementedError('This function must be implemented by the student.')
    return expr
    

    

def softmax(x, derivative = False):
    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis = 1, keepdims = True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis = 1, keepdims = True)))
    else:
        return softmax(x) * (1 - softmax(x))

"""
def cross_entropy(y, p):
    
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.

    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.

    
    
    #raise NotImplementedError('This function must be implemented by the student.')
    float32_epsilon = np.finfo(np.float32).eps
    norm_loss = -np.sum(y * np.log(p)) / float(p.shape[0])
    loss = -np.sum(y * np.log(p)) + float32_epsilon
    return
"""


def cross_entropy(y, p):
    """
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.

    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.
    """
    
    #raise NotImplementedError('This function must be implemented by the student.')
    num_y_samples = y.shape[0]
    pred = softmax(p)
    
    log_prob = -np.log(pred[range(num_y_samples), y])
    loss = np.sum(log_prob) / num_y_samples
    return loss


def one_hot_encoding(y):
    """
    Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot
    encoding creates new binary-valued columns, each of which indicate the presence of each possible value from the
    original data.

    Args:
        y: A numpy array of shape (n_samples,) representing the target class values for each sample in the input data.

    Returns:
        A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input
        data. n_outputs is equal to the number of unique categorical class values in the numpy array y.
    """
    
    shape = (y.size, y.max()+1)
    one_hot = np.zeros(shape)
    
    rows = np.arange(y.size)
    
    one_hot[rows, y] = 1
    
    return one_hot
    
    #raise NotImplementedError('This function must be implemented by the student.')

    
