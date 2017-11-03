# Create model for training the CIFAR-10 data set. 
# There is already a model created in TensorFlow

import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import theano 
#import tensorflow as tf
from get_input import get_train_hyperparameters, do_unpickle

class model(object):
    def __init__(self):
        # initialize the hyperparameters
        self.X_tr, self.Y_tr = get_train_hyperparameters(flatten = False)
        # initialize more hyperparameters like the stride length, batch size, etc. 
        
if __name__ == '__main__':
    CIFARMODEL = model()
    print(CIFARMODEL.X_tr.shape)
    print(CIFARMODEL.Y_tr.shape)


