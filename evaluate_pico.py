import os
import glob
import time
import argparse

# Kerasa / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from loss import depth_loss_function
from utils import predict, load_images, display_images, evaluate
from matplotlib import pyplot as plt
import numpy as np
import cv2

def load_groundtruth(path):
    depths = []
    for imfile in sorted(glob.glob(os.path.join(path, "*.png"))):
        img = cv2.imread(imfile, cv2.IMREAD_ANYDEPTH)
        depths.append(img)

    inds = np.arange(len(depths)).tolist()
    depths = np.array([depths[i] for i in inds]) / 1000
    return depths

def load_test_images(path):
    images = []
    for imfile in sorted(glob.glob(os.path.join(path, "*.png"))):
        img = cv2.imread(imfile)
        images.append(img)
    inds = np.arange(len(images)).tolist()
    images = [images[i] for i in inds]
    images = np.stack(images).astype(np.float32)
    return images
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
print('Loading test data...', end='')
#depth = load_images( glob.glob('data/pico/gt/*.png') )
depth = load_groundtruth('data/pico/gt')
#rgb = load_test_images('data/pico/rgb')
rgb = load_images(sorted(glob.glob('data/pico/rgb/*.png')))
print(depth.shape)
print(rgb.shape)
crop = [20, 459, 24, 615]

print('Test data loaded.\n')

start = time.time()
print('Testing...')

e = evaluate(model, rgb, depth, crop, batch_size=6)

print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5]))

end = time.time()
print('\nTest time', end-start, 's')

