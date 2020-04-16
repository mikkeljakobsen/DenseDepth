import os
import glob
import time
import argparse
from utils import load_void_test_data, load_test_data

# Kerasa / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from loss import depth_loss_function
from utils import predict, load_images, display_images, evaluate
from matplotlib import pyplot as plt
import numpy as np

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--dataset', default='nyu', type=str, help='Test dataset.')
parser.add_argument('--use-median-scaling', dest='use_median_scaling', action='store_true', help='If true, all predictions are scaled by median gt before evaluation.')
parser.add_argument('--use-sparse-depth', dest='use_sparse_depth', action='store_true', help='If true, all predictions are scaled by median sparse depth before evaluation.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}

# Load model into GPU / CPU
print('Loading model...')
model = load_model(args.model, custom_objects=custom_objects, compile=False)

# Load test data
print('Loading test data...', end='')
test_set = {}
if(args.dataset == 'nyu'):
	test_set = load_test_data()
else:
	test_set = load_void_test_data()
print('Test data loaded.\n')

start = time.time()
print('Testing...')
if args.use_sparse_depth:
	e = evaluate(model, test_set['rgb'], test_set['depth'], test_set['crop'], batch_size=6, verbose=True, use_median_scaling=True, interp_depth=test_set['interp_depth'])
elif args.use_median_scaling:
	e = evaluate(model, test_set['rgb'], test_set['depth'], test_set['crop'], batch_size=6, verbose=True, use_median_scaling=True)
else:
	e = evaluate(model, test_set['rgb'], test_set['depth'], test_set['crop'], batch_size=6, verbose=True, use_median_scaling=False)
#e = evaluate(model, rgb, depth, crop, batch_size=6)

end = time.time()
print('\nTest time', end-start, 's')
