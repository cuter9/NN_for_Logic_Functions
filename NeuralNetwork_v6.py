# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 18:12:25 2018

@author: cuterhsu

1. add bias weight of layers
2. add activation function
3. add network configuration parameter, nemLayer, numInput, and numOutput  and deepness of NN
4. add learning rate and momentum rate
5. add Levenberg-Marquardt algorithm
6. implemented by using pytorch

"""

import numpy as np
# from numpy import exp, array, random, dot, tanh
from numpy import exp, random, tanh
# import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import os
import time
import math
from training_algorithms import *


class NeuralNetwork:
    def __init__(self, **nn_params):
        # self.device = torch.device('cuda')

        random.seed(1)  # Can be any integer between 0 and 2**32 - 1 inclusive
        # We initialize our weights randomely and uniformly with a half open interval [0, 1].
        # Though we want it to be [-1, 1].
        self.network_params = nn_params['network_params']
        self.train_params = nn_params['train_params']
        self.samples = nn_params['samples']

        self.numInput = self.network_params['numInputs']
        self.numHdLayers = self.network_params['numHdLayers']
        self.numOutput = self.network_params['numOutputs']
        self.deepness = len(self.numHdLayers)  # number of layers, excluding input layer
        self.Layers = self.numInput + self.numHdLayers + self.numOutput
        self.bias = self.network_params['bias']
        self.struc_nn = ''
        self.weight_shapes = []
        self.bias_shapes = []
        self.W = []
        self.b = []

        self.lr = self.train_params['lr']
        self.mem = self.train_params['mem']
        self.train_method = self.train_params['train_method']
        self.epochs = self.train_params['epochs']
        self.goal = self.train_params['perf_goal']
        self.mu_pwin = 10  # the window for evaluating the mu adjustment
        self.mu_dr = self.train_params['mu_dr']
        self.mu_ir = self.train_params['mu_ir']
        self.mu = self.train_params['mu']

        self.inputs = self.samples['inputs']
        self.targets = self.samples['targets']
        self.samples_size = self.samples['samples_size']
        self.batch_size = self.samples['batch_size']
        self.total_batch = self.samples_size // self.batch_size
        self.total_batch = self.total_batch if (self.samples_size % self.batch_size == 0) else self.total_batch + 1

        self.trainlm = trainlm
        self.traingdm = traingdm

    # sigmoid activation for binary classification
    @staticmethod
    def perceptron(x, W, b):
        xh = x @ W + b
        yh = torch.sigmoid(xh)
        return yh

    @staticmethod
    def sigmoid(x):
        yh = 1 / (1 + exp(-x))
        yh_grad = yh * (1 - yh)
        return yh, yh_grad

    @staticmethod
    def softmax(x):
        yh = exp(-x)
        yh = yh / np.sum(yh, axis=1)
        yh_grad = - np.identity(np.size(yh, axis=1))
        return yh, yh_grad

    @staticmethod
    def hypertangent(x):
        return tanh(x)

    def initialization(self):
        self.struc_nn = str().join([str(a) for a in self.Layers])
        self.weight_shapes = [(a, b) for a, b in zip(self.Layers[0:-1], self.Layers[1:])]
        self.bias_shapes = [(1, b) for b in self.Layers[1:]]
        self.W = [torch.tensor(2 * (2 * np.random.random(s) - 1), requires_grad=True) for s in
                  self.weight_shapes]
        self.b = [torch.tensor(2 * (2 * np.random.random((1, s)) - 1), requires_grad=True) for s in
                  self.Layers[1:]]

    # train network with training method
    #    def train(self, inputs, targets):
    def train(self):
        targets_t = torch.tensor(self.targets, requires_grad=True)
        inputs_t = torch.tensor(self.inputs, requires_grad=True)
        targets_b = torch.split(targets_t, self.batch_size)
        inputs_b = torch.split(inputs_t, self.batch_size)

        match self.train_method:
            case 'lm':
                perf = self.trainlm(self, inputs_b, targets_b)
            case 'gdm':
                perf = self.traingdm(self, inputs_b, targets_b)
        return perf

    # the neurons of the hidden and output layer.
    # We need both to later backpropagate
    def inference(self, inputs):
        # yh = [inputs]
        self.yh = [inputs]  # [[yh(0)]]

        for W, b, b_shape in zip(self.W, self.b, self.bias_shapes):
            yh = self.perceptron(self.yh[-1], W, b)
            self.yh.append(yh)  # [[yh(0)] [yh(1)..sel...]]

        outputs = self.yh[-1]
        return outputs

    def inference_softmax(self, inputs):

        yh_t = inputs
        self.yh = []  # [[yh(0)]]
        self.yh_grad = []
        idx_layer = 1
        for W, b in zip(self.W, self.b):
            # for n in range(self.deepness):
            if idx_layer == self.numHdLayers:  # softmax for output layer
                yh_t, yh_grad_t = self.activation_softmax(yh_t @ W + b.T)
            else:  # sigmoid for hidden layer
                yh_t, yh_grad_t = self.activation(yh_t @ W + b.T)
                idx_layer += 1
            self.yh.append(yh_t)  # [[yh(0)] [yh(1).....]]
            self.yh_grad.append(yh_grad_t)  # [[yh(0)] [yh(1).....]]
        self.output = self.yh[-1]

        # @staticmethod

    def activation(self, x):
        yh, yh_grad = self.sigmoid(x)
        return yh, yh_grad

    def activation_softmax(self, x):
        yh, yh_grad = self.softmax(x)
        return yh, yh_grad

        # same as __inference but just returns the output

    def think(self, inputs):
        inputs = torch.tensor(inputs)
        outputs = self.inference(inputs)
        return outputs
