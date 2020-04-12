# Original Matlab code https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
#
#
# Python port of depth filling code from NYU toolbox
# Speed needs to be improved
#
# Uses 'pypardiso' solver 
#
import scipy
import skimage
import numpy as np
from pypardiso import spsolve
from PIL import Image
from skimage.color import rgb2gray
import os
from scipy.interpolate import LinearNDInterpolator
import os, os.path
import errno

# Taken from https://stackoverflow.com/a/600612/119527
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

#
# fill_depth_colorization.m
# Preprocesses the kinect depth image using a gray scale version of the
# RGB image as a weighting for the smoothing. This code is a slight
# adaptation of Anat Levin's colorization code:
#
# See: www.cs.huji.ac.il/~yweiss/Colorization/
#
# Args:
#  imgRgb - HxWx3 matrix, the rgb image for the current frame. This must
#      be between 0 and 1.
#  imgDepth - HxW matrix, the depth image for the current frame in
#       absolute (meters) space.
#  alpha - a penalty value between 0 and 1 for the current depth values.
def interpolate_depth(depth_map, validity_map, log_space=False):
  '''
  Interpolate sparse depth with barycentric coordinates
  Args:
    depth_map : np.float32
      H x W depth map
    validity_map : np.float32
      H x W depth map
    log_space : bool
      if set then produce in log space
  Returns:
    np.float32 : H x W interpolated depth map
  '''
  assert depth_map.ndim == 2 and validity_map.ndim == 2
  rows, cols = depth_map.shape
  data_row_idx, data_col_idx = np.where(validity_map)
  depth_values = depth_map[data_row_idx, data_col_idx]
  # Perform linear interpolation in log space
  if log_space:
    depth_values = np.log(depth_values)
  interpolator = LinearNDInterpolator(
      # points=Delaunay(np.stack([data_row_idx, data_col_idx], axis=1).astype(np.float32)),
      points=np.stack([data_row_idx, data_col_idx], axis=1),
      values=depth_values,
      fill_value=0 if not log_space else np.log(1e-3))
  query_row_idx, query_col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
  query_coord = np.stack([query_row_idx.ravel(), query_col_idx.ravel()], axis=1)
  Z = interpolator(query_coord).reshape([rows, cols])
  if log_space:
    Z = np.exp(Z)
    Z[Z < 1e-1] = 0.0
  return Z

def save_depth(z, path):
	'''
	Saves a depth map to a 16-bit PNG file
	Args:
	z : numpy
	  depth map
	path : str
	  path to store depth map
	'''
	z = np.uint32(z*256.0)
	z = Image.fromarray(z, mode='I')
    if not os.path.isfile(path):
        mkdir_p(os.path.dirname(path))
	z.save(path)

if __name__ == "__main__":
	void_train_depth = list(line.strip() for line in open('/home/mikkel/void_150/train_ground_truth.txt'))
	void_test_depth = list(line.strip() for line in open('/home/mikkel/void_150/test_ground_truth.txt'))

#	for i in range(0, len(void_train_depth)):
#		y = np.asarray(np.asarray(Image.open( "/home/mikkel/"+void_train_depth[i] ))/256.0)
#		y[y <= 0] = 0.0
#		y[y > 10] = 0.0
#		v = y.astype(np.float32)
#		v[y > 0] = 1.0
#		v[y > 10] = 0.0
#		y = interpolate_depth(y, v) # fill missing pixels and convert to cm		
#		save_depth(y, "/home/mikkel/new_void/"+void_train_depth[i])
#		print("Processed train image", i, "out of", len(void_train_depth))


	for i in range(0, len(void_test_depth)):
		y = np.asarray(np.asarray(Image.open( "/home/mikkel/"+void_test_depth[i] ))/256.0)
		y[y <= 0] = 0.0
		y[y > 10] = 0.0
		v = y.astype(np.float32)
		v[y > 0] = 1.0
		v[y > 10] = 0.0
		y = interpolate_depth(y, v) # fill missing pixels and convert to cm		
		save_depth(y, "/home/mikkel/new_void/"+void_test_depth[i])
		print("Processed test image", i, "out of", len(void_test_depth))
