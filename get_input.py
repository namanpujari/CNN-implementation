import _pickle as cPickle
import os 
import sys
import numpy as np
import time

def do_unpickle(file_name):
	with open('./cifar-10-batches-py/'+file_name, 'rb') as to_open:
            dict = cPickle.load(to_open, encoding = 'bytes')
	return dict

def square_resize_channels(arr, start_ind=0, end_ind=3072, size=32):
    '''
    Resizes a flattened array from start_ind, to end_ind, 
    into a size * size matrix given the criteria are fit. 
    '''
    master_arr = np.zeros([3, size, size], dtype = int)
    if(end_ind - start_ind != np.square(size) * 3):
        sys.exit('Unable to square flattened array of specified length')
    # If it works then we know length of flat array is 1024.
    for channel in [0, 1, 2]:
        channel_flat_arr_start = start_ind + (channel * 1024)
        channel_flat_arr_end = start_ind + ((channel + 1) * 1024)
        channel_flat_arr = arr[channel_flat_arr_start:channel_flat_arr_end]
        for row in range(size):
            start_splice = channel_flat_arr_start + (size * row) # Point at which we splice the row
            master_arr[channel][row, :] = arr[start_splice:(start_splice + (size))]
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

    Xtrain = np.zeros([50000, 3, 32, 32])
    # Complete packaging of the first batch
    for index, batch in enumerate(list(data.keys())[0:1]):  
        raw_segment = np.array(data[batch][b'data'])
        for image in range(index*10000, 10000*(index+1), 1):
            image_data = raw_segment[image, :]
            image_data_resized = square_resize_channels(image_data)
            for channel in range(0, 3, 1):
                #print('in batch ' + batch + ' in image ' + str(image) + ' in channel ' + str(channel))
                Xtrain[image][channel] = image_data_resized[0]

                #square_arr_to_append = raw_segment[image][channel*1024:((channel+1)*1024)]
                #print(square_arr_to_append.shape)
                #Xtrain[image][channel] = square_resize(square_arr_to_append, channel*1024, 
                #                                           (channel+1)*1024) 

    '''
    for index, batch in enumerate(data.keys()):
        if():
            X_train = np.array(data[batch][b'data'])
            Y_train = np.array(data[batch][b'labels'])
            Y_train = Y_train.reshape((Y_train.shape[0], 1)) 
        else:
            X_train_segment = np.array(data[batch][b'data'])
            Y_train_segment = np.array(data[batch][b'labels'])
            Y_train_segment = Y_train_segment.reshape((Y_train_segment.shape[0], 1))
            X_train = np.append(X_train, X_train_segment, axis = 0)
            Y_train = np.append(Y_train, Y_train_segment, axis = 0)
    if(flatten == False): 
        return X_train.reshape((X_train.shape[0], 32, 32, 3)), Y_train 
    else: 
        return X_train, Y_train
    '''

    return Xtrain