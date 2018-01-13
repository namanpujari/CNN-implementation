import _pickle as cPickle
import os 
import sys
import numpy as np
import time

def do_unpickle(file_name):
	with open('./cifar-10-batches-py/'+file_name, 'rb') as to_open:
            dict = cPickle.load(to_open, encoding = 'bytes')
	return dict

def resize(arr, start_ind=0, end_ind=3072):
    '''
    Resizes a flattened array from start_ind, to end_ind, 
    into a size * size matrix given the criteria are fit. 
    '''
    red_channel_start = 0
    green_channel_start = 1024
    blue_channel_start = 2048
    master_arr = np.zeros([32, 32, 3], dtype = int)
    if(end_ind - start_ind != np.square(32) * 3):
        sys.exit('Unable to square flattened array of specified length')
    # If it works then we know length of flat array is 3072.
    for row in range(32):
        # An array of the start indices for each channel (red = 0, green = 1, blue = 2)
        channel_start_inds = [red_channel_start + (row * 32), green_channel_start + (row * 32), 
                                blue_channel_start + (row * 32)]
        for column in range(32):
            column_array = [arr[channel_start_inds[0] + column], arr[channel_start_inds[1] + column], 
                            arr[channel_start_inds[2] + column]]
            master_arr[row][column] = np.array(column_array)    
    return master_arr

def get_train_hyperparameters(flatten=False):
    data = {}
    files = os.listdir('./cifar-10-batches-py')
    file_count = 0;
    for file_index in range(len(files)):
        if(str(files[file_index]).startswith('data_batch')):
            file_count = file_count + 1	
            data["batch_{}".format(str(file_count))] = do_unpickle(files[file_index]) 
    # The code above this point unwraps the binary data into ndarrays stored within dictionaries.
    # We may now aggregate all the ndarray's for training into a simple ndarray

    # The variable data is a dictionary holding the batches, which are in turn dictionaries
    # see README_UNPICKLE for details on what the batch dictionaries contain.

    # According to CIFAR, the data is stored in 10000 * 3072 format, with each row 
    # being an image. The first 1024 values in the row are red channel values in 
    # ROW MAJOR ORDER; that is, the first 32 values from the red channel values
    # are those of the FIRST ROW of the image.

    # I am sort of doubtful as to how numpy handles the resizing of the 10000 * 3072
    # ndarray, so I will do it myself. 

    Xtrain = np.zeros([50000, 32, 32, 3])
    # Complete packaging of the first batch
    for index, batch in enumerate(list(data.keys())[0:1]):  
        raw_segment = np.array(data[batch][b'data'])
        for image in range(0, 10000, 1):
            image_data = raw_segment[image, :]
            image_data_resized = resize(image_data)
            Xtrain[(10000 * index) + image] = image_data_resized

    return Xtrain