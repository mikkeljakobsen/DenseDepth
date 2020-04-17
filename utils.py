import numpy as np
from PIL import Image

def DepthNorm(x, maxDepth):
    return maxDepth / x

def predict(model, images, minDepth=10, maxDepth=1000, batch_size=2):
    # Support multiple RGBs, one RGB image, even grayscale 
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

def load_void_test_data(void_data_path='/home/mikkel/data/void_release', use_sparse_depth=False):
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
            img = np.asarray(Image.open( os.path.join("/home/mikkel/data/void_sparse/", interp_depth_path).replace('ground_truth', 'interp_depth') ))/256.0
            interp_depths.append(img)
        inds = np.arange(len(interp_depths)).tolist()
        interp_depths = np.array([interp_depths[i] for i in inds])
    return {'rgb':images, 'depth':depths, 'interp_depth':interp_depths, 'crop': [20, 459, 24, 615]}#[0, 480, 0, 640]}

def compute_errors(gt, pred, min_depth=0.1, max_depth=10.0):
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
    return a1, a2, a3, abs_rel, rmse, log_10

def evaluate(model, rgb, depth, crop, batch_size=6, verbose=False, use_median_scaling=False, interp_depth=None):
    N = len(rgb)

    bs = batch_size

    predictions = []
    testSetDepths = []
    imu_depths = []
    
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

        imu_depth = []
        if interp_depth is not None:
            imu_depth = interp_depth[(i)*bs:(i+1)*bs,:,:]
            imu_depth = imu_depth[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        
        # Compute errors per image in batch
        for j in range(len(true_y)):
            prediction = (0.5 * pred_y[j]) + (0.5 * np.fliplr(pred_y_flip[j]))
            if use_median_scaling:
                if interp_depth is not None:
                    predictions.append(prediction*compute_scaling_factor(imu_depth[j], prediction))
                else:
                    predictions.append(prediction*compute_scaling_factor(true_y[j], prediction))
            else:
                predictions.append(prediction)
            testSetDepths.append(   true_y[j]   )

    predictions = np.stack(predictions, axis=0)
    testSetDepths = np.stack(testSetDepths, axis=0)

    e = compute_errors(testSetDepths, predictions)

    if verbose:
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0],e[1],e[2],e[3],e[4],e[5]))

    return e
