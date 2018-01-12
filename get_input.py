import _pickle as cPickle
import os 
import sys
import numpy as np

def do_unpickle(file_name):
	with open('./cifar-10-batches-py/'+file_name, 'rb') as to_open:
            dict = cPickle.load(to_open, encoding = 'bytes')
	return dict

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
    for batch in data.keys():
        if(batch == 'batch_1'):
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
