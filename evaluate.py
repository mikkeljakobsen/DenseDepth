import os
import glob
import time
import argparse
from utils import load_void_imu_test_data, load_void_test_data, load_test_data, load_void_rgb_sparse_test_data, load_void_pred_sparse_test_data
# Kerasa / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from loss import depth_loss_function
from utils import predict, load_images, display_images, evaluate, evaluate_rgb_sparse, evaluate_pred_sparse
from matplotlib import pyplot as plt
import numpy as np

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')
parser.add_argument('--dataset', default='void', type=str, help='Test dataset.')
parser.add_argument('--channels', default=3, type=int, help='Number of channels for VOID dataset.')
parser.add_argument('--use-median-scaling', default=False, dest='use_median_scaling', action='store_true', help='If true, all predictions are scaled by median gt before evaluation.')
parser.add_argument('--use-sparse-depth-scaling', default=False, dest='use_sparse_depth_scaling', action='store_true', help='If true, all predictions are scaled by median sparse depth before evaluation.')
parser.add_argument('--dont-interpolate', default=False, dest='dont_interpolate', action='store_true', help='Use raw sparse depth maps for refinement (dont interpolate).')
parser.add_argument('--use-scaling-array', default=False, dest='use_scaling_array', action='store_true', help='If true, all predictions are scaled by a scaling array before evaluation.')
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
elif(args.dataset == 'void-imu'):
	test_set = load_void_imu_test_data()
elif(args.dataset == 'void-rgb-sparse'):
	test_set = load_void_rgb_sparse_test_data()
elif(args.dataset == 'void-pred-sparse'):
	test_set = load_void_pred_sparse_test_data()
else:
	test_set = load_void_test_data(use_sparse_depth=args.use_sparse_depth_scaling, dont_interpolate=args.dont_interpolate, channels=args.channels)
print('Test data loaded.\n')

start = time.time()
print('Testing...')
if args.dataset == 'void-rgb-sparse':
	e = evaluate_rgb_sparse(model, test_set['rgb'], test_set['sparse_depth_and_vm'], test_set['depth'], test_set['crop'], batch_size=6, verbose=True, use_median_scaling=args.use_median_scaling)
elif(args.dataset == 'void-pred-sparse'):
	e = evaluate_pred_sparse(model, test_set['init_preds'], test_set['sparse_depths'], test_set['depth'], test_set['crop'], batch_size=6, verbose=True, use_median_scaling=args.use_median_scaling)
elif args.use_sparse_depth_scaling:
	e = evaluate(model, test_set['rgb'], test_set['depth'], test_set['crop'], batch_size=6, verbose=True, use_median_scaling=True, interp_depth=test_set['interp_depth'], use_scaling_array=args.use_scaling_array)
elif args.use_median_scaling:
	e = evaluate(model, test_set['rgb'], test_set['depth'], test_set['crop'], batch_size=6, verbose=True, use_median_scaling=True)
else:
	e = evaluate(model, test_set['rgb'], test_set['depth'], test_set['crop'], batch_size=6, verbose=True, use_median_scaling=False)
#e = evaluate(model, rgb, depth, crop, batch_size=6)

end = time.time()
print('\nTest time', end-start, 's')
