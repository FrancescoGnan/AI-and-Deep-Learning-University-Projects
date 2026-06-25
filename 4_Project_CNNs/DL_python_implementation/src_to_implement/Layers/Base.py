import numpy as np
import matplotlib.pyplot as plt

# 2. Base Layer
class BaseLayer:
    # this class is a parent class for all layers

    def __init__(self):
        self.trainable = False
        self.weights = None
