# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 18:18:44 2020

@author: cuterhsu

1. implemented by using pytorch
"""
import numpy as np
from NeuralNetwork_v6 import NeuralNetwork
from numpy import array
import time

# the parameters of the network
numInputs = [3]  # full adder inputs
numHdLayers = [10, 10]  # [number of layer nodes, ..., number of layer nodes]
numOutputs = [2]        # full adder outputs
bias = 1
network_params = {'numInputs': numInputs, 'numHdLayers': numHdLayers, 'numOutputs': numOutputs, 'bias': bias}

# parameters for the network training
perf_goal = 10e-13  # training performance in MSE
no_epochs = 2000  # training epochs
lr = 0.1  # learning rate
mem = 0.98  # momentum rate
mu = 10.0  # initial mu for LM method
mu_ir = 2.0  # increment rate for adjusting mu
mu_dr = 0.7  # decrement rate for adjusting mu
train_method = 'lm'  # 'gdm' for gradient descent w momentum; 'lm' for lm
train_params = {'train_method': train_method, 'perf_goal': perf_goal, 'epochs': no_epochs, 'mem': mem, 'lr': lr,
                'mu': mu, 'mu_ir': mu_ir, 'mu_dr': mu_dr}

# training data for full adder simulation,  Xfd : input, Yfd : target
Xi = array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]], dtype=float)
Yi = array([[0, 0],
             [0, 1],
             [0, 1],
             [1, 0],
             [0, 1],
             [1, 0],
             [1, 0],
             [1, 1]], dtype=float)


# parameters of training sample
samples_size = {'samples_size': Xi.shape[0]}  # the number of samples for training
batch_size = {'batch_size': Xi.shape[0]}  # the batch size of  training samples per in a epoch
# batch_size = {'batch_size' : 1}
samples = {**samples_size, **batch_size, 'inputs': Xi, 'targets': Yi}  # parameters of training samples

# combine all parameters of the neural network
NN_Params = {'samples': samples, 'train_params': train_params, 'network_params': network_params}


# With the above parameters 'NN_Params',
# set up the Neural Networks configuration using a class 'NeuralNetwork'  in a module 'NN'.
# You are to design a python module named 'NN' with a class named 'NeuralNetwork' to simulate as required in homework


def main():
    nn = NeuralNetwork(**NN_Params)
    nn.initialization()
    perf = nn.train()

    # start training
    start_train = time.time()
    end_train = time.time()
    print('training time :', end_train - start_train)  # calculate the time in training

    # the training convergence and data are to be returned through dictionary key 'perf_epoch' in terms of epoch
    perf_epoch = perf['perf_epoch']
    stop_epoch = perf['stop_epoch']  # the epoch that the training stops in any case
    print("\n The training error : ", perf_epoch[stop_epoch], "at", stop_epoch + 1, "th epoch \n")

    return nn


def nn_eval(nn):
    # test the simulation performance of network after training with a function think in NeuralNetwork
    np.set_printoptions(precision=4)  # set precision for displaying data
    for n in range(8):
        print('simulation result for x({0})= {1} : {2} '.format(n, Xi[n, :], nn.think(Xi[n])))


if __name__ == "__main__":
    network = main()
    nn_eval(network)
