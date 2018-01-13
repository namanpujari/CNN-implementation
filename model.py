# Create model for training the CIFAR-10 data set. 
# There is already a model created in TensorFlow

import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
#import tensorflow as tf
from get_input import get_train_hyperparameters, do_unpickle
from PIL import Image

class model(object):
    def __init__(self):
        # initialize the hyperparameters
        #self.X_tr, self.Y_tr = get_train_hyperparameters(flatten = False)
        self.X_tr = get_train_hyperparameters(flatten = False).astype('uint8')
        self.labels = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
        # initialize more ihyperparameters like the stride length, batch size, etc. 
        
if __name__ == '__main__':
    CIFARMODEL = model()

    image_ind = 120;
    print('This should be a 32x32 image of a bird.')
    img = Image.fromarray(CIFARMODEL.X_tr[image_ind])
    img.save('test.png')

