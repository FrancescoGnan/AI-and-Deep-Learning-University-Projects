# Deep Learning -- Assignment 1; Jonah & Ekki

import numpy as np
import copy
import matplotlib.pyplot as plt
import functools

# 7. Neural Network Skeleton
class NeuralNetwork:

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None

    def forward(self):
        # fetch input data and save the label tensor for backward
        (tensor, self.label_tensor) = self.data_layer.next()
        # propagate through all layers
        #tensor = functools.reduce(lambda t, layer: layer.forward(t), self.layers, tensor)
        for layer in self.layers:
            tensor = layer.forward(tensor)
        # return output of loss layer
        return self.loss_layer.forward(tensor, self.label_tensor)

    def backward(self):
        # compute starting error_tensor from loss layer
        tensor = self.loss_layer.backward(self.label_tensor)
        # traverse the layers in reverse
        #tensor = functools.reduce(lambda t, layer: layer.forward(t), reversed(self.layers), tensor)
        for layer in reversed(self.layers):
            tensor = layer.backward(tensor)

    def append_layer(self, layer):
        # deep copy optimizer to trainable layer (don't want references to another object (have own state))
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        # append layer regardless of trainability
        self.layers.append(layer)

    def train(self, iterations):
        # train the network iterations
        #def _train_step(self):
        #    self.loss.append(self.forward())
        #    self.backward()
        #functools.reduce(lambda _, __: _train_step(), range(iterations), None)
        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        # pass the input tensor through all layers
        tensor = input_tensor
        #tensor = functools.reduce(lambda t, layer: layer.forward(t), self.layers, input_tensor)
        for layer in self.layers:
            tensor = layer.forward(tensor)
        return tensor