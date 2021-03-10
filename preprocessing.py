from PIL import Image
import numpy as np
import os
import  PIL
import glob
import warnings
from skimage.io import imread, imsave
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import keras



def aspect_ratio_resize_smart(img,base=256): #img is a PIL image variable
	'''
	:param img: the image
	:param base: the base, default 256
	:return:
	this function aims to perform an hight resoltion of size of images
	'''
	if img.size[0] <= img.size[1]:
		basewidth = base
		wpercent = (basewidth/float(img.size[0]))
		hsize = int((float(img.size[1])*float(wpercent)))
		img = img.resize((basewidth,hsize),PIL.Image.LANCZOS)
	else:
		baseheight = base
		wpercent = (baseheight/float(img.size[1]))
		wsize = int((float(img.size[0])*float(wpercent)))
		img = img.resize((wsize,baseheight),PIL.Image.LANCZOS)
	return img



def read_img_from_location(location):
	"""

	:param location:  the loation
	:return: the image arrays , x and y array fro trainign or evaluation

	read image from location. It take teh image and transform it into accurate size for our task. Thus it read image and
	take from it its three channel pixels and the its label. Finally, return the two array for training or evaluation.

	"""
	x_train = []
	y_train = []
	dirs = os.listdir(location)
	label = 0
	for i in dirs:
		for pic in glob.glob(location+i+'/*.png'): #in format .png (all image for reading dataset)
			im = Image.open(pic)
			im = aspect_ratio_resize_smart(im)
			im = np.array(im)
			if(im.shape[0]==256 and im.shape[1] ==256):
				r = im[:,:,0]
				g = im[:,:,1]
				b = im[:,:,2]
				x_train.append([r,g,b])
				y_train.append([label])
		label = label + 1
	return np.array(x_train),np.array(y_train)


def data_normaization(x,y):
	"""

	:param x:  the array x
	:param y:  the array y (labels)
	:return:  the array x and y normalized

	function that perform an normalization of data
	"""
	x_data = x.astype('float32')

	x_data /= 255

	y_data = keras.utils.to_categorical(y, 21)
	return  x_data,y_data



def create_train_test_validate_dataset(location):
	"""

	:param location: the location (path)
	:return: the dirs of new folder created (train/test/validation)

	This funtion create the test, validation and train dataset from dataset UCMerced original if they doesnt exist.
	If they alreasy exist, then it return the target dirs from dir which is called "flow".

	"""

	# Collect class names from directory names in './data/UCMerced_LandUse/Images/'
	class_names = os.listdir(location)

	# Create path to image "flow" base directory
	flow_base = os.path.join('data', 'flow')

	# Create pathnames to train/validate/test subdirectories
	target_dirs = {target: os.path.join(flow_base, target) for target in ['train', 'validate', 'test']}

	if not os.path.isdir(flow_base):

		# Make new directories
		os.mkdir(flow_base)

		for target in ['train', 'validate', 'test']:
			target_dir = os.path.join(flow_base, target)
			os.mkdir(target_dir)
			for class_name in class_names:
				class_subdir = os.path.join(target_dir, class_name)
				os.mkdir(class_subdir)

		# suppress low-contrast warning from skimage.io.imsave
		warnings.simplefilter('ignore', UserWarning)

		# Copy images from ./data/UCMerced_LandUse/Images to ./data/flow/<train, validate, test>
		for root, _, filenames in os.walk(location):
			if filenames:
				class_name = os.path.basename(root)

				# Randomly shuffle filenames
				filenames = np.random.permutation(filenames)
				for target, count in [('train', 80), ('validate', 10), ('test', 10)]:
					target_dir = os.path.join(flow_base, target, class_name)
					for filename in filenames[:count]:
						filepath = os.path.join(root, filename)
						image = imread(filepath)
						basename, _ = os.path.splitext(filename)
						# Convert TIF to PNG to work with Keras ImageDataGenerator.flow_from_directory
						target_filename = os.path.join(target_dir, basename + '.png')
						imsave(target_filename, image)

					filenames = filenames[count:]

		# Show future warnings during development
		warnings.resetwarnings()

	return  target_dirs

def get_bottleneck_features(model, dataset, preproc_func, target_dirs,batch_size=64):
	"""
	:param model: the model
	:param dataset: the dataset
	:param preproc_func: the preprocessing funtion
	:param target_dirs: teh targte dirs
	:param batch_size: teh batch size (default 64)
	:return:
	'''This funtion aims to get botteleneck features X and labels Y for the input dataset (train/validate/test)
	by predicting on the convolutional portion only of a pretrained model.

	Note: Saves features and labels to numpy files for future use when rerunning the code.

	Inputs:
	model: Pre-trained deep learning model, excluding fully-connected top model e.g. applications.VGG16(include_top=False, weights='imagenet')
	dataset = string label for dataset image directory ['train', 'validate', 'test']
	preproc_func: preprocessing function implied to each input sample
	batch_size: number of image samples per batch

	Return: Return bottleneck features as numpy.array
    """
	print(f'Generating "{dataset}" bottleneck predictions')
	X_filepath = ".\\data\\bottleneck_features\\bn_" + dataset + "_X.npy"
	y_filepath = ".\\data\\bottleneck_features\\bn_" + dataset + "_y.npy"

	# Check if data are available from disk.
	try:
		with open(X_filepath, 'rb') as f:
			X = np.load(f)
		with open(y_filepath, 'rb') as f:
			y = np.load(f)
			Y = to_categorical(y, num_classes=len(np.unique(y)))
	# Else, get the bottleneck features and labels
	except:
		image_data_gen = ImageDataGenerator(rescale=1/255.0, preprocessing_function=preproc_func)
		image_generator = image_data_gen.flow_from_directory(target_dirs[dataset],batch_size=batch_size,shuffle=False)
		image_count = 0
		X_batches, Y_batches = [], []
		for X, Y in image_generator:
			X_batches.append(model.predict_on_batch(X))
			Y_batches.append(Y)
			image_count += X.shape[0]
			# Must interrupt image_generator
			if image_count >= image_generator.n:
				break

		X = np.concatenate(X_batches)
		with open(X_filepath, 'wb') as f:
			np.save(f, X)
		Y = np.concatenate(Y_batches)
		y = np.nonzero(Y)[1]
		with open(y_filepath, 'wb') as f:
			np.save(f, y)

	print(f'   Features of shape {X.shape} extracted for model "{model.name}"')
	return X, Y






