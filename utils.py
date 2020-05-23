import numpy as np
from PIL import Image
import os
from scipy.interpolate import LinearNDInterpolator
import settings
import glob

def nyu_resize(img, resolution=480, padding=6):
    from skimage.transform import resize
    return resize(img, (resolution, int(resolution*4/3)), preserve_range=True, mode='reflect', anti_aliasing=True )

def DepthNorm(x, maxDepth):
    if settings.USE_DEPTHNORM: 
        return maxDepth / x
    else: 
        return x

def predict(model, images, minDepth=settings.MIN_DEPTH*settings.DEPTH_SCALE, maxDepth=settings.MAX_DEPTH*settings.DEPTH_SCALE, batch_size=2):
    # Support multiple RGBs, one RGB image, even grayscale 
    if isinstance(images, list):
        #print("rgb", images[0].shape, "sparse:", images[1].shape)
        for i in range(len(images)):
            if len(images[i].shape) < 4: images[i] = images[i].reshape((1, images[i].shape[0], images[i].shape[1], images[i].shape[2]))
    else:
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

    jet = plt.get_cmap('jet')

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
            imgs.append(jet(rescaled)[:,:,:3])
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

def compute_scaling_factor(gt, pr, min_depth=settings.MIN_DEPTH, max_depth=settings.MAX_DEPTH):
    gt = np.array(gt, dtype=np.float64).reshape(-1)
    pr = np.array(pr, dtype=np.float64).reshape(-1)
    # only use valid depth values
    v = (gt > min_depth) & (gt < max_depth)
    return np.median(gt[v] / pr[v])

def compute_scaling_array(gt, pr, min_depth=settings.MIN_DEPTH, max_depth=settings.MAX_DEPTH):
    gt = np.array(gt, dtype=np.float64)
    pr = np.array(pr, dtype=np.float64)
    rows, cols = gt.shape
    v = (gt > min_depth) & (gt < max_depth)# & (np.random.random(gt.shape) < 0.05)
    data_row_idx, data_col_idx = np.where(v)
    scale_values = gt[data_row_idx, data_col_idx] / pr[data_row_idx, data_col_idx]
    # Perform linear interpolation in log space
    interpolator = LinearNDInterpolator(
        points=np.stack([data_row_idx, data_col_idx], axis=1),
        values=scale_values,
        fill_value=1.0)
    query_row_idx, query_col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    query_coord = np.stack([query_row_idx.ravel(), query_col_idx.ravel()], axis=1)
    return interpolator(query_coord).reshape([rows, cols])

def load_custom_test_data(path, channels=3, use_sparse_depth=False, dont_interpolate=False, gt_divider=1000.0, use_void_1500=False):
    images, depths, interp_depths = [], [], []
    suffix = ''
    #if use_void_1500: suffix = '_1500'
    for rgb_path in sorted(glob.glob(os.path.join(path, 'interp_depth'+suffix+'/*.png'))):
        img = np.asarray(Image.open( rgb_path.replace('interp_depth'+suffix, 'image') )).reshape(480,640,3)
        if channels > 3:
            if dont_interpolate:
                iz = DepthNorm(np.clip(np.asarray(Image.open( rgb_path.replace('interp_depth', 'sparse_depth') ))/256.0*settings.DEPTH_SCALE, settings.MIN_DEPTH*settings.DEPTH_SCALE, settings.MAX_DEPTH*settings.DEPTH_SCALE), maxDepth=settings.MAX_DEPTH*settings.DEPTH_SCALE)*255
            else:
                iz = DepthNorm(np.clip(np.asarray(Image.open( rgb_path ))/256.0*settings.DEPTH_SCALE, settings.MIN_DEPTH*settings.DEPTH_SCALE, settings.MAX_DEPTH*settings.DEPTH_SCALE), maxDepth=settings.MAX_DEPTH*settings.DEPTH_SCALE)*255
            if channels > 4:
                vm = np.array(Image.open(rgb_path.replace('interp_depth', 'validity_map')), dtype=np.float32)*255
                img = np.stack([img[:,:,0], img[:,:,1], img[:,:,2], iz, vm], axis=-1)
            else:
                img = np.stack([img[:,:,0], img[:,:,1], img[:,:,2], iz], axis=-1)
        depth = np.asarray(Image.open( rgb_path.replace('interp_depth'+suffix, 'ground_truth') ), dtype=np.float32)/gt_divider # convert from mm to m
        images.append(img)
        depths.append(depth)

        if use_sparse_depth:
            if dont_interpolate: interp_depth = np.asarray(Image.open( rgb_path.replace('interp_depth', 'sparse_depth') ))/256.0
            else: interp_depth = np.asarray(Image.open( rgb_path ))/256.0
            interp_depths.append(interp_depth)

    inds = np.arange(len(images)).tolist()
    images = [images[i] for i in inds]
    images = np.stack(images).astype(np.float32)
    depths = np.array([depths[i] for i in inds])
    if use_sparse_depth: interp_depths = np.array([interp_depths[i] for i in inds])
    return {'rgb':images, 'depth':depths, 'interp_depth':interp_depths, 'crop': [20, 459, 24, 615]}
    

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

def load_void_test_data(void_data_path='/home/mikkel/data/void_release', channels=3, use_sparse_depth=False, dont_interpolate=False, use_void_1500=False):
    if use_void_1500:
        void_test_rgb_temp = list(line.strip() for line in open(void_data_path+'/void_1500/test_image.txt'))
        void_test_depth_temp = list(line.strip().replace('ground_truth', 'ground_truth_orig') for line in open(void_data_path+'/void_1500/test_ground_truth.txt'))
    else:
        void_test_rgb_temp = list(line.strip() for line in open(void_data_path+'/void_150/test_image.txt'))
        void_test_depth_temp = list(line.strip() for line in open(void_data_path+'/void_150/test_ground_truth.txt'))
    void_test_rgb, void_test_depth = [], []
    for i in range(len(void_test_rgb_temp)):
        if os.path.isfile(os.path.join(void_data_path, void_test_rgb_temp[i]).replace('image', 'sparse_depth')):
            void_test_rgb.append(void_test_rgb_temp[i])
            void_test_depth.append(void_test_depth_temp[i])
    print('test size:', len(void_test_rgb))
    images = []

    for rgb_path in void_test_rgb:
        img = np.asarray(Image.open( void_data_path+"/"+rgb_path )).reshape(480,640,3)
        if channels > 3:
            #iz = np.clip(np.asarray(Image.open( os.path.join(void_data_path, rgb_path).replace('image', 'interp_depth')))/256.0/10.0,0,1)*255
            if dont_interpolate:
                iz = DepthNorm(np.clip(np.asarray(Image.open( os.path.join(void_data_path, rgb_path).replace('image', 'sparse_depth') ))/256.0*settings.DEPTH_SCALE, settings.MIN_DEPTH*settings.DEPTH_SCALE, settings.MAX_DEPTH*settings.DEPTH_SCALE), maxDepth=settings.MAX_DEPTH*settings.DEPTH_SCALE)*255
            else:
                iz = DepthNorm(np.clip(np.asarray(Image.open( os.path.join(void_data_path, rgb_path).replace('image', 'interp_depth') ))/256.0*settings.DEPTH_SCALE, settings.MIN_DEPTH*settings.DEPTH_SCALE, settings.MAX_DEPTH*settings.DEPTH_SCALE), maxDepth=settings.MAX_DEPTH*settings.DEPTH_SCALE)*255
            if channels > 4:
                vm = np.array(Image.open(os.path.join(void_data_path, rgb_path).replace('image', 'validity_map')), dtype=np.float32)
                assert(np.all(np.unique(vm) == [0, 256]))
                vm[vm > 0] = 255
                img = np.stack([img[:,:,0], img[:,:,1], img[:,:,2], iz, vm], axis=-1)
            else:
                img = np.stack([img[:,:,0], img[:,:,1], img[:,:,2], iz], axis=-1)
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
        for interp_depth_path in void_test_rgb:
            if dont_interpolate: img = np.asarray(Image.open( os.path.join(void_data_path, interp_depth_path).replace('image', 'sparse_depth') ))/256.0
            else: img = np.asarray(Image.open( os.path.join(void_data_path, interp_depth_path).replace('image', 'interp_depth') ))/256.0
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
        x2 = np.clip(np.asarray(Image.open( os.path.join(void_data_path, rgb_path).replace('image', 'prediction')))/256.0/settings.MAX_DEPTH,0,1).reshape(480,640)*255
        x3 = np.clip(np.asarray(Image.open( os.path.join(void_data_path, rgb_path).replace('image', 'interp_depth')))/256.0/settings.MAX_DEPTH,0,1).reshape(480,640)*255
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
        iz = np.clip(np.asarray(Image.open( os.path.join(void_data_path, rgb_path).replace('image', 'interp_depth') ))/256.0,0,1)*255
        vm = np.array(Image.open(os.path.join(void_data_path, rgb_path).replace('image', 'validity_map')), dtype=np.float32)*255
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
        init_pred = nyu_resize(DepthNorm(np.clip(np.asarray(Image.open( os.path.join(void_data_path, rgb_path).replace('image', 'prediction') ))/256.0*settings.DEPTH_SCALE,settings.MIN_DEPTH*settings.DEPTH_SCALE,settings.MAX_DEPTH*settings.DEPTH_SCALE).reshape(480,640,1), maxDepth=settings.MAX_DEPTH*settings.DEPTH_SCALE), 240)*255
        sparse_depth = nyu_resize(DepthNorm(np.clip(np.asarray(Image.open( os.path.join(void_data_path, rgb_path).replace('image', 'interp_depth') ))/256.0*settings.DEPTH_SCALE,settings.MIN_DEPTH*settings.DEPTH_SCALE,settings.MAX_DEPTH*settings.DEPTH_SCALE).reshape(480,640,1), maxDepth=settings.MAX_DEPTH*settings.DEPTH_SCALE), 240)*255
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

def compute_errors(gt, pred, min_depth=settings.MIN_DEPTH_EVAL, max_depth=settings.MAX_DEPTH_EVAL):
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
    log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    mae = np.mean(np.abs(gt - pred))
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    i_mae = np.mean(np.abs(1.0/gt - 1.0/pred))
    i_rmse = np.sqrt(np.mean((1.0/gt - 1.0/pred)**2))

    return a1, a2, a3, abs_rel, rmse, log_10, scinv, mae, i_mae, i_rmse, rmse_log

def save_img(image, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    image.save(path)

def evaluate(model, rgb, depth, crop, batch_size=6, verbose=False, use_median_scaling=False, interp_depth=None, use_scaling_array=False, save_pred=False, model_name='output'):
    N = len(rgb)
    count = 0
    bs = batch_size

    predictions = []
    testSetDepths = []
    
    for i in range(N//bs):    
        x = rgb[(i)*bs:(i+1)*bs,:,:,:]

        # Compute results
        true_y = depth[(i)*bs:(i+1)*bs,:,:]
        pred_y = scale_up(2, predict(model, x/255, minDepth=settings.MIN_DEPTH*settings.DEPTH_SCALE, maxDepth=settings.MAX_DEPTH*settings.DEPTH_SCALE, batch_size=bs)[:,:,:,0]) * settings.MAX_DEPTH
        
        # Test time augmentation: mirror image estimate
        pred_y_flip = scale_up(2, predict(model, x[...,::-1,:]/255, minDepth=settings.MIN_DEPTH*settings.DEPTH_SCALE, maxDepth=settings.MAX_DEPTH*settings.DEPTH_SCALE, batch_size=bs)[:,:,:,0]) * settings.MAX_DEPTH

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
                    if use_scaling_array: 
                        scale_array = compute_scaling_array(sparse_depth[j], prediction)
                        #scale_array = compute_scaling_array(true_y[j], prediction)
                        prediction = prediction*scale_array
                        print("interp depth average scale", np.mean(scale_array))
                    else: 
                        scale = compute_scaling_factor(sparse_depth[j], prediction)
                        prediction = prediction*scale                   
                        print("sparse depth scaling factor", scale)
                else:
                    scale = compute_scaling_factor(true_y[j], prediction)
                    prediction = prediction*scale
                    print("scaling factor", scale)
            predictions.append(prediction)
            testSetDepths.append(true_y[j])
            if save_pred:
                import matplotlib.pyplot as plt
                from skimage.transform import resize
                from scipy import ndimage
                path = "/home/mikkel/samples/" + model_name + "/pred_viz_all/" + str(count) +".png"
                #print(path)
                count = count+1
                save_img(Image.fromarray(np.uint32(prediction.copy()*256.0), mode='I'), path.replace('pred_viz_all','pred_raw_depth'))
                jet = plt.get_cmap('jet')
                jet.set_bad(color='black')
                h, w = true_y[j].shape[0], true_y[j].shape[1]
                #rgb = resize(x[j,:,:,:3], (h,w), preserve_range=True, mode='reflect', anti_aliasing=True)
                gt = np.clip(true_y[j].copy(), 0.0, settings.MAX_DEPTH_EVAL)/settings.MAX_DEPTH_EVAL
                gt[gt==0.0] = np.nan
                gt = jet(gt)[:,:,:3]
                pr = np.clip(prediction.copy(), 0.0, settings.MAX_DEPTH_EVAL)/settings.MAX_DEPTH_EVAL
                pr[pr==0.0] = np.nan
                pr = jet(pr)[:,:,:3]

                if interp_depth is not None:
                    sd = np.clip(sparse_depth[j].copy(), 0.0, settings.MAX_DEPTH_EVAL)/settings.MAX_DEPTH_EVAL
                    sd = ndimage.grey_dilation(np.uint8(sd*255), size=(3, 3)) / 255.0
                    sd[sd==0.0] = np.nan
                    sd = jet(sd)[:,:,:3]
                    sd = sd*255
                    sd = Image.fromarray(np.uint8(sd))                    
                    save_img(sd, path.replace('pred_viz_all','pred_viz_interp_depth'))
                    
                #pr = jet(predict(model, x[j]/255, minDepth=settings.MIN_DEPTH*settings.DEPTH_SCALE, maxDepth=settings.MAX_DEPTH*settings.DEPTH_SCALE)[0,:,:,0])[:,:,:3]
                #pr = resize(pr, (x[j].shape[0], x[j].shape[1]), order=1, preserve_range=True, mode='reflect', anti_aliasing=True )
                #pr = pr[crop[0]:crop[1]+1, crop[2]:crop[3]+1]
                img = x[j,crop[0]:crop[1]+1, crop[2]:crop[3]+1,:3].copy()
                img = resize(img, (h, w), preserve_range=True, mode='reflect', anti_aliasing=True )
                gt, pr = gt*255, pr*255
                output_img = np.vstack([img, gt, pr])
                height, width, channel = output_img.shape
                save_img(Image.fromarray(np.uint8(output_img)), path)
                save_img(Image.fromarray(np.uint8(pr)), path.replace('pred_viz_all','pred_viz_depth'))
                save_img(Image.fromarray(np.uint8(gt)), path.replace('pred_viz_all','pred_viz_gt'))
                save_img(Image.fromarray(np.uint8(img)), path.replace('pred_viz_all','orig_image'))
        #print("tested", (i+1)*bs, "out of", N, "test images")

    #predictions = np.stack(predictions, axis=0)
    #testSetDepths = np.stack(testSetDepths, axis=0)

    e = []
    for i in range(len(predictions)): e.append(compute_errors(testSetDepths[i], predictions[i]))
    e = np.array(e).mean(0)
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
        pred_y = scale_up(2, predict(model, x/255, minDepth=settings.MIN_DEPTH*settings.DEPTH_SCALE, maxDepth=settings.MAX_DEPTH*settings.DEPTH_SCALE, batch_size=bs)[:,:,:,0]) * settings.MAX_DEPTH
        
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
        pred_y = scale_up(2, predict(model, x/255, minDepth=settings.MIN_DEPTH*settings.DEPTH_SCALE, maxDepth=settings.MAX_DEPTH*settings.DEPTH_SCALE, batch_size=bs)[:,:,:,0]) * settings.MAX_DEPTH
        
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