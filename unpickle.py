import _pickle as cPickle
import os 
import sys
import numpy as np

def do_unpickle(file_name):
	with open('./cifar-10-batches-py/'+file_name, 'rb') as to_open:
            dict = cPickle.load(to_open, encoding = 'bytes')
	return dict

def get_hyperparameters():
    data = {}
    files = os.listdir('./cifar-10-batches-py')
    file_count = 0;
    for file_index in range(len(files)):
        if(str(files[file_index]).startswith('data_batch')):
            file_count = file_count + 1	
            data["batch_{}".format(str(file_count))] = do_unpickle(files[file_index])
    
    # print(data['batch_1'][b'data'].shape)
    #print(data.keys())

    # The code above this point unwraps the binary data into ndarrays stored within dictionaries.
    # We may now aggregate all the ndarray's for training into a simple ndarray
    
    for batch in data.keys():
        #print('in ' + str(batch));
        #print('size of its data is ' + str(data[batch][b'data'].shape))
        if(batch == 'batch_1'):
            X_train = np.array(data[batch][b'data'])
            Y_train = np.array(data[batch][b'labels'])
            Y_train = Y_train.reshape((Y_train.shape[0], 1)) 
            #print('size of its labels is ' + str(Y_train.shape))
        else:
            X_train_segment = np.array(data[batch][b'data'])
            Y_train_segment = np.array(data[batch][b'labels'])
            Y_train_segment = Y_train_segment.reshape((Y_train_segment.shape[0], 1))
            X_train = np.append(X_train, X_train_segment, axis = 0)
            Y_train = np.append(Y_train, Y_train_segment, axis = 0)
            #print('size of its labels is ' + str(Y_train_segment.shape))

    return X_train, Y_train
    #print('the total size of X_train is now ' + str(X_train.shape))
    #print('the total size of Y_train is now ' + str(Y_train.shape))

if __name__ == '__main__':
    X, Y = get_hyperparameters();
    print(X.shape)
    print(Y.shape)













        
        
        
        
      
