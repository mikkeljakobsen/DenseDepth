import numpy as np
from PIL import Image
import os
from scipy.interpolate import LinearNDInterpolator

def nyu_resize(img, resolution=480, padding=6):
    from skimage.transform import resize
    return resize(img, (resolution, int(resolution*4/3)), preserve_range=True, mode='reflect', anti_aliasing=True )

def DepthNorm(x, maxDepth):
    return maxDepth / x

def predict(model, images, minDepth=10, maxDepth=1000, batch_size=2):
    # Support multiple RGBs, one RGB image, even grayscale 
    if not isinstance(images, list):
        #print("rgb", images[0].shape, "sparse:", images[1].shape)
        #if len(images[0].shape) < 4: images = [images[0].reshape((1, images[0].shape[0], images[0].shape[1], images[0].shape[2])), images[1].reshape((1, images[1].shape[0], images[1].shape[1], images[1].shape[2]))]
    #else:
        #print(images.shape)
        if len(images.shape) < 3: images = np.stack((images,images,images), axis=2)
        if len(images.shape) < 4: images = images.reshape((1, images.shape[0], images.shape[1], images.shape[2]))
    # Compute predictions
    predictions = model.predict(images, batch_size=batch_size)
    # Put in expected range
    return np.clip(DepthNorm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth

def scale_up(scale, images):
    from skimage.transform import resize
    scaled = []
    
    for i in range(len(images)):
        img = images[i]
        output_shape = (scale * img.shape[0], scale * img.shape[1])
        scaled.append( resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True ) )

    return np.stack(scaled)

def load_images(image_files):
    loaded_images = []
    for file in image_files:
        x = np.clip(np.asarray(Image.open( file ), dtype=float) / 255, 0, 1)
        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)

def to_multichannel(i):
    if i.shape[2] == 3: return i
    i = i[:,:,0]
    return np.stack((i,i,i), axis=2)
        
def display_images(outputs, inputs=None, gt=None, is_colormap=True, is_rescale=True):
    import matplotlib.pyplot as plt
    import skimage
    from skimage.transform import resize

    plasma = plt.get_cmap('plasma')

    shape = (outputs[0].shape[0], outputs[0].shape[1], 3)
    
    all_images = []

    for i in range(outputs.shape[0]):
        imgs = []
        
        if isinstance(inputs, (list, tuple, np.ndarray)):
            x = to_multichannel(inputs[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True )
            imgs.append(x)

        if isinstance(gt, (list, tuple, np.ndarray)):
            x = to_multichannel(gt[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True )
            imgs.append(x)

        if is_colormap:
            rescaled = outputs[i][:,:,0]
            if is_rescale:
                rescaled = rescaled - np.min(rescaled)
                rescaled = rescaled / np.max(rescaled)
            imgs.append(plasma(rescaled)[:,:,:3])
        else:
            imgs.append(to_multichannel(outputs[i]))

        img_set = np.hstack(imgs)
        all_images.append(img_set)

    all_images = np.stack(all_images)
    
    return skimage.util.montage(all_images, multichannel=True, fill=(0,0,0))

def save_images(filename, outputs, inputs=None, gt=None, is_colormap=True, is_rescale=False):
    montage =  display_images(outputs, inputs, is_colormap, is_rescale)
    im = Image.fromarray(np.uint8(montage*255))
    im.save(filename)

def compute_scaling_factor(gt, pr, min_depth=0.1, max_depth=10.0):
    gt = np.array(gt, dtype=np.float64).reshape(-1)
    pr = np.array(pr, dtype=np.float64).reshape(-1)

    # only use valid depth values
    v = (gt > min_depth) & (gt < max_depth)
    return np.median(gt[v] / pr[v])

def compute_scaling_array(gt, pr, min_depth=0.1, max_depth=10.0):
    gt = np.array(gt, dtype=np.float64)
    pr = np.array(pr, dtype=np.float64)
    rows, cols = gt.shape
    data_row_idx, data_col_idx = np.where((gt > min_depth) & (gt < max_depth))
    scale_values = gt[data_row_idx, data_col_idx] / pr[data_row_idx, data_col_idx]
    # Perform linear interpolation in log space
    interpolator = LinearNDInterpolator(
        points=np.stack([data_row_idx, data_col_idx], axis=1),
        values=scale_values,
        fill_value=1.0)
    query_row_idx, query_col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    query_coord = np.stack([query_row_idx.ravel(), query_col_idx.ravel()], axis=1)
    return interpolator(query_coord).reshape([rows, cols])

def load_test_data(test_data_zip_file='nyu_test.zip'):
    print('Loading test data...', end='')
    import numpy as np
    from data import extract_zip
    data = extract_zip(test_data_zip_file)
    from io import BytesIO
    rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
    depth = np.load(BytesIO(data['eigen_test_depth.npy']))
    crop = np.load(BytesIO(data['eigen_test_crop.npy']))
    print('Test data loaded.\n')
    return {'rgb':rgb, 'depth':depth, 'crop':crop}

def load_void_test_data(void_data_path='/home/mikkel/data/void_release', use_sparse_depth=False, dont_interpolate=False):
    void_test_rgb = list(line.strip() for line in open(void_data_path+'/void_150/test_image.txt'))
    void_test_depth = list(line.strip() for line in open(void_data_path+'/void_150/test_ground_truth.txt'))

    images = []
    for rgb_path in void_test_rgb:
        img = np.asarray(Image.open( void_data_path+"/"+rgb_path )).reshape(480,640,3)
        images.append(img)
    inds = np.arange(len(images)).tolist()
    images = [images[i] for i in inds]
    images = np.stack(images).astype(np.float32)

    depths = []
    for depth_path in void_test_depth:
        img = np.asarray(Image.open( void_data_path+"/"+depth_path ))/256.0
        depths.append(img)
    inds = np.arange(len(depths)).tolist()
    depths = np.array([depths[i] for i in inds])

    interp_depths = []
    if use_sparse_depth:
        for interp_depth_path in void_test_depth:
            if dont_interpolate: img = np.asarray(Image.open( os.path.join(void_data_path, interp_depth_path).replace('ground_truth', 'sparse_depth') ))/256.0
            else: img = np.asarray(Image.open( os.path.join(void_data_path, interp_depth_path).replace('ground_truth', 'interp_depth') ))/256.0
            interp_depths.append(img)
        inds = np.arange(len(interp_depths)).tolist()
        interp_depths = np.array([interp_depths[i] for i in inds])
    return {'rgb':images, 'depth':depths, 'interp_depth':interp_depths, 'crop': [20, 459, 24, 615]}#[0, 480, 0, 640]}

def load_void_imu_test_data(void_data_path='/home/mikkel/data/void_release'):
    void_test_rgb = list(line.strip() for line in open(void_data_path+'/void_150/test_image.txt'))
    void_test_depth = list(line.strip() for line in open(void_data_path+'/void_150/test_ground_truth.txt'))

    images = []
    for rgb_path in void_test_rgb:
        x1 = np.asarray(Image.open( os.path.join(void_data_path, rgb_path)).convert('L')).reshape(480,640)
        x2 = np.clip(np.asarray(Image.open( os.path.join(void_data_path, rgb_path).replace('image', 'prediction')))/256.0/10.0,0,1).reshape(480,640)*255
        x3 = np.clip(np.asarray(Image.open( os.path.join(void_data_path, rgb_path).replace('image', 'interp_depth')))/256.0/10.0,0,1).reshape(480,640)*255
        images.append(np.stack([x1, x2, x3], axis=-1))
    inds = np.arange(len(images)).tolist()
    images = [images[i] for i in inds]
    images = np.stack(images).astype(np.float32)

    depths = []
    for depth_path in void_test_depth:
        img = np.asarray(Image.open( void_data_path+"/"+depth_path ))/256.0
        depths.append(img)
    inds = np.arange(len(depths)).tolist()
    depths = np.array([depths[i] for i in inds])   
    return {'rgb':images, 'depth':depths, 'crop': [20, 459, 24, 615]}#[0, 480, 0, 640]}

def load_void_rgb_sparse_test_data(void_data_path='/home/mikkel/data/void_release'):
    void_test_rgb = list(line.strip() for line in open(void_data_path+'/void_150/test_image.txt'))
    void_test_depth = list(line.strip() for line in open(void_data_path+'/void_150/test_ground_truth.txt'))

    images = []
    sparse_depth_and_vm = []
    for rgb_path in void_test_rgb:
        im = np.asarray(Image.open( os.path.join(void_data_path, rgb_path)) ).reshape(480,640,3)
        iz = np.clip(np.asarray(Image.open( os.path.join(void_data_path, rgb_path).replace('image', 'interp_depth') ))/256.0/10.0,0,1).reshape(480,640)*255
        vm = np.array(Image.open(os.path.join(void_data_path, rgb_path).replace('image', 'validity_map')), dtype=np.float32).reshape(480,640)*255
        images.append(im)
        sparse_depth_and_vm.append(np.stack([iz, vm], axis=-1))
    inds = np.arange(len(images)).tolist()
    images = [images[i] for i in inds]
    images = np.stack(images).astype(np.float32)
    sparse_depth_and_vm = [sparse_depth_and_vm[i] for i in inds]
    sparse_depth_and_vm = np.stack(sparse_depth_and_vm).astype(np.float32)

    depths = []
    for depth_path in void_test_depth:
        img = np.asarray(Image.open( void_data_path+"/"+depth_path ))/256.0
        depths.append(img)
    inds = np.arange(len(depths)).tolist()
    depths = np.array([depths[i] for i in inds])   
    return {'rgb':images, 'depth':depths, 'sparse_depth_and_vm': sparse_depth_and_vm, 'crop': [20, 459, 24, 615]}#[0, 480, 0, 640]}

def load_void_pred_sparse_test_data(void_data_path='/home/mikkel/data/void_release'):
    void_test_rgb = list(line.strip() for line in open(void_data_path+'/void_150/test_image.txt'))
    void_test_depth = list(line.strip() for line in open(void_data_path+'/void_150/test_ground_truth.txt'))

    init_preds = []
    sparse_depths = []
    for rgb_path in void_test_rgb:
        init_pred = nyu_resize(DepthNorm(np.clip(np.asarray(Image.open( os.path.join(void_data_path, rgb_path).replace('image', 'prediction') ))/256.0*100,10.0,1000.0).reshape(480,640,1), maxDepth=1000), 240)*255
        sparse_depth = nyu_resize(DepthNorm(np.clip(np.asarray(Image.open( os.path.join(void_data_path, rgb_path).replace('image', 'interp_depth') ))/256.0*100,10.0,1000.0).reshape(480,640,1), maxDepth=1000), 240)*255
        init_preds.append(init_pred)
        sparse_depths.append(sparse_depth)
    inds = np.arange(len(init_preds)).tolist()
    init_preds = [init_preds[i] for i in inds]
    init_preds = np.stack(init_preds).astype(np.float32)
    sparse_depths = [sparse_depths[i] for i in inds]
    sparse_depths = np.stack(sparse_depths).astype(np.float32)

    depths = []
    for depth_path in void_test_depth:
        img = np.asarray(Image.open( void_data_path+"/"+depth_path ))/256.0
        depths.append(img)
    inds = np.arange(len(depths)).tolist()
    depths = np.array([depths[i] for i in inds])   
    return {'init_preds':init_preds, 'sparse_depths': sparse_depths, 'depth':depths, 'crop': [20, 459, 24, 615]}#[0, 480, 0, 640]}

# copied from https://github.com/lmb-freiburg/demon
def scale_invariant(gt, pr):
    """
    Computes the scale invariant loss based on differences of logs of depth maps.
    Takes preprocessed depths (no nans, infs and non-positive values)
    depth1:  one depth map
    depth2:  another depth map
    Returns:
        scale_invariant_distance
    """
    gt = gt.reshape(-1)
    pr = pr.reshape(-1)

    v = gt > 0.1
    gt = gt[v]
    pr = pr[v]

    log_diff = np.log(gt) - np.log(pr)
    num_pixels = np.float32(log_diff.size)

    # sqrt(Eq. 3)
    return np.sqrt(np.sum(np.square(log_diff)) / num_pixels \
        - np.square(np.sum(log_diff)) / np.square(num_pixels))

def compute_errors(gt, pred, min_depth=0.1, max_depth=10.0):
    if isinstance(pred, list):
        scinv_list = []
        for i in range(len(gt)):
            scinv_list.append(scale_invariant(gt[i], pred[i]))
        scinv = np.mean(scinv_list)
        gt = np.stack(gt).astype(np.float32).reshape(-1)
        pred = np.stack(pred).astype(np.float32).reshape(-1)
    else:
        scinv = scale_invariant(gt, pred)
    # igore invalid depth values from evaluation
    v = (gt > min_depth) & (gt < max_depth)
    gt, pred = gt[v], pred[v]
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()

    mae = np.mean(np.abs(gt - pred))
    i_mae = np.mean(np.abs(1.0/gt - 1.0/pred))
    i_rmse = np.sqrt(np.mean((1.0/gt - 1.0/pred)**2))
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    return a1, a2, a3, abs_rel, rmse, log_10, scinv, mae, i_mae, i_rmse, rmse_log

def evaluate(model, rgb, depth, crop, batch_size=6, verbose=False, use_median_scaling=False, interp_depth=None, use_scaling_array=False):
    N = len(rgb)

    bs = batch_size

    predictions = []
    testSetDepths = []
    
    for i in range(N//bs):    
        x = rgb[(i)*bs:(i+1)*bs,:,:,:]

        # Compute results
        true_y = depth[(i)*bs:(i+1)*bs,:,:]
        pred_y = scale_up(2, predict(model, x/255, minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0
        
        # Test time augmentation: mirror image estimate
        pred_y_flip = scale_up(2, predict(model, x[...,::-1,:]/255, minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0

        # Crop based on Eigen et al. crop
        true_y = true_y[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_y = pred_y[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_y_flip = pred_y_flip[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]

        sparse_depth = []
        if interp_depth is not None:
            sparse_depth = interp_depth[(i)*bs:(i+1)*bs,:,:]
            sparse_depth = sparse_depth[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        
        # Compute errors per image in batch
        for j in range(len(true_y)):
            prediction = (0.5 * pred_y[j]) + (0.5 * np.fliplr(pred_y_flip[j]))
            if use_median_scaling:
                if interp_depth is not None:
                    if use_scaling_array: predictions.append(prediction*compute_scaling_array(sparse_depth[j], prediction))
                    else: predictions.append(prediction*compute_scaling_factor(sparse_depth[j], prediction))
                else:
                    predictions.append(prediction*compute_scaling_factor(true_y[j], prediction))
            else:
                predictions.append(prediction)
            testSetDepths.append(   true_y[j]   )
        print("tested", (i+1)*bs, "out of", N, "test images")

    predictions = np.stack(predictions, axis=0)
    testSetDepths = np.stack(testSetDepths, axis=0)

    e = compute_errors(testSetDepths, predictions)

    if verbose:
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rmse', 'log_10', 'scinv', 'mae', 'i_mae', 'i_rmse', 'rmse_log'))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5],e[6],e[7],e[8],e[9],e[10]))

    return e

def evaluate_rgb_sparse(model, rgb, sparse_depth_and_vm, depth, crop, batch_size=6, verbose=False, use_median_scaling=False):
    N = len(rgb)
    N_sparse = len(sparse_depth_and_vm)
    print(N, 'rgb images.', N_sparse, 'sparse+vm images')

    bs = batch_size

    predictions = []
    testSetDepths = []
    
    for i in range(N//bs):    
        x = rgb[(i)*bs:(i+1)*bs,:,:,:]
        iz_and_vm = sparse_depth_and_vm[(i)*bs:(i+1)*bs,:,:,:]

        # Compute results
        true_y = depth[(i)*bs:(i+1)*bs,:,:]
        pred_y = scale_up(2, predict(model, [x/255, iz_and_vm/255], minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0
        
        # Test time augmentation: mirror image estimate
        #pred_y_flip = scale_up(2, predict(model, [x[...,::-1,:]/255, iz_and_vm[...,::-1,:]/255], minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0

        # Crop based on Eigen et al. crop
        true_y = true_y[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_y = pred_y[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        #pred_y_flip = pred_y_flip[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        
        # Compute errors per image in batch
        for j in range(len(true_y)):
            #prediction = (0.5 * pred_y[j]) + (0.5 * np.fliplr(pred_y_flip[j]))
            if use_median_scaling:
                pred_y[j] = pred_y[j]*compute_scaling_factor(true_y[j], pred_y[j])
            predictions.append(pred_y[j])
            testSetDepths.append(true_y[j])
        print("tested", (i+1)*bs, "out of", N, "test images")
    predictions = np.stack(predictions, axis=0)
    testSetDepths = np.stack(testSetDepths, axis=0)

    e = compute_errors(testSetDepths, predictions)

    if verbose:
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rmse', 'log_10', 'scinv', 'mae', 'i_mae', 'i_rmse', 'rmse_log'))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5],e[6],e[7],e[8],e[9],e[10]))

    return e

def evaluate_pred_sparse(model, init_preds, sparse_depths, depth, crop, batch_size=6, verbose=False, use_median_scaling=False):
    N = len(init_preds)

    bs = batch_size

    predictions = []
    testSetDepths = []
    
    for i in range(N//bs):    
        x = init_preds[(i)*bs:(i+1)*bs,:,:]
        iz = sparse_depths[(i)*bs:(i+1)*bs,:,:]

        # Compute results
        true_y = depth[(i)*bs:(i+1)*bs,:,:]
        pred_y = scale_up(2, predict(model, [x/255, iz/255], minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0
        
        # Test time augmentation: mirror image estimate
        #pred_y_flip = scale_up(2, predict(model, [x[...,::-1,:]/255, iz_and_vm[...,::-1,:]/255], minDepth=10, maxDepth=1000, batch_size=bs)[:,:,:,0]) * 10.0

        # Crop based on Eigen et al. crop
        true_y = true_y[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_y = pred_y[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        #pred_y_flip = pred_y_flip[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        
        # Compute errors per image in batch
        for j in range(len(true_y)):
            #prediction = (0.5 * pred_y[j]) + (0.5 * np.fliplr(pred_y_flip[j]))
            if use_median_scaling:
                pred_y[j] = pred_y[j]*compute_scaling_factor(true_y[j], pred_y[j])
            predictions.append(pred_y[j])
            testSetDepths.append(true_y[j])
        print("tested", (i+1)*bs, "out of", N, "test images")

    predictions = np.stack(predictions, axis=0)
    testSetDepths = np.stack(testSetDepths, axis=0)

    e = compute_errors(testSetDepths, predictions)

    if verbose:
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rmse', 'log_10', 'scinv', 'mae', 'i_mae', 'i_rmse', 'rmse_log'))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5],e[6],e[7],e[8],e[9],e[10]))

    return e