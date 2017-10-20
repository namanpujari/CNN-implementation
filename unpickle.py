import cPickle
import os 
import sys

def do_unpickle(file_name):
	with open('./cifar-10-batches-py/'+file_name, 'rb') as to_open:
		dict = cPickle.load(to_open)
	return dict

if __name__ == "__main__":
	data = {}
	files = os.listdir('./cifar-10-batches-py')
	file_count = 0;
	for file_index in range(len(files)):
		if(str(files[file_index]).startswith('data_batch')):
			file_count = file_count+1	
			data["batch_{}".format(str(file_count))] = do_unpickle(files[file_index])	

	# instructions on how the script works
	with open('./README_UNPICKLE', 'wb') as f:
		f.write('Reaid CIFAR-10 database into dictionary\n')
		f.write('The keys of the dictionary are\n\n')
                for item in data.keys():
                    if(str(item) != 'batch_5'):
                        f.write(str(item) + ' , ')
                    else:
                        f.write(str(item))
                f.write('\n')
		#f.write(data.keys())
		f.write('\nWithin each key is another dictionary that holds the following values for each data batch:\n\n')
                for item in data['batch_1'].keys():
                    if(str(item) != 'filenames'):
                        f.write(str(item) + ' , ')
                    else:
                        f.write(str(item))
                f.write('\n')
		#f.write(data['batch_1'].keys())
		f.write('\nThe data key holds a numpy.ndarray of 8 bit unsigned ints indicated RGB values of a 32x32 image\n')
		f.write('Here is some sample data:')
		#f.write(data['batch_1']['data'])	

