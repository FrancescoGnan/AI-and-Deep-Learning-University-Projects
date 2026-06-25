# Deep Learning -- Assignment 1; Jonah & Ekki

import numpy as np
import matplotlib.pyplot as plt

# 1. Basic Optimizer
class Sgd:

    # constructor
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    # method: calc. update
    def calculate_update(self, weight_tensor, gradient_tensor):
        # see formula in slide set: w(k+1) = w(k) - eta * gradient
        return weight_tensor - self.learning_rate * gradient_tensor