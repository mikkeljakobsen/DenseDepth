import os
import glob
import time
import argparse
from PIL import Image

# Kerasa / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from loss import depth_loss_function
from utils import predict
from matplotlib import pyplot as plt
import numpy as np


# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}

# Load model into GPU / CPU
print('Loading model...')
model = load_model(args.model, custom_objects=custom_objects, compile=False)

# Load test data
print('Loading list of train and test data...', end='')
void_train_rgb = list("/home/mikkel/data/void_release/"+line.strip() for line in open('/home/mikkel/data/void_release/void_150/train_image.txt'))
void_test_rgb = list("/home/mikkel/data/void_release/"+line.strip() for line in open('/home/mikkel/data/void_release/void_150/test_image.txt'))
void_rgb_list = void_train_rgb + void_test_rgb

start = time.time()
print('Predicting depth...')

N = len(void_rgb_list)

bs = 6

predictions = []

for i in range(N//bs):
	x = []
	rgb_paths = void_rgb_list[(i)*bs:(i+1)*bs]
	for rgb_path in (void_train_rgb+void_test_rgb):
		img = np.asarray(Image.open( rgb_path )).reshape(480,640,3)
		x.append(img)
	inds = np.arange(len(x)).tolist()
	x = [x[i] for i in inds]
	x = np.stack(x).astype(np.float32)

	# Compute results
	pred_y = scale_up(2, predict(model, x/255, minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0

	# Test time augmentation: mirror image estimate
pred_y_flip = scale_up(2, predict(model, x[...,::-1,:]/255, minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0
	
	# Compute errors per image in batch
	for j in range(len(x)):
		path = rgb_paths[j].replace('image', 'prediction')
		prediction = (0.5 * pred_y[j]) + (0.5 * np.fliplr(pred_y_flip[j]))
		z = np.uint32(prediction*256.0)
		z = Image.fromarray(z, mode='I')
		if not os.path.exists(path):
			os.makedirs(path)
		z.save(path)
	print('Saved', i*bs, 'out of', N)


end = time.time()
print('\nTotal time', end-start, 's')
