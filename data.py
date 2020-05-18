import numpy as np
from utils import DepthNorm
from io import BytesIO
from PIL import Image
from zipfile import ZipFile
from keras.utils import Sequence
from augment import BasicPolicy
import os
from fill_depth_colorization import interpolate_depth
import settings

def extract_zip(input_zip):
    input_zip=ZipFile(input_zip)
    return {name: input_zip.read(name) for name in input_zip.namelist()}

def nyu_resize(img, resolution=480, padding=6):
    from skimage.transform import resize
    return resize(img, (resolution, int(resolution*4/3)), preserve_range=True, mode='reflect', anti_aliasing=True )

def get_nyu_data(batch_size, nyu_data_zipfile='nyu_data.zip'):
    data = extract_zip(nyu_data_zipfile)

    nyu2_train = list((row.split(',') for row in (data['data/nyu2_train.csv']).decode("utf-8").split('\n') if len(row) > 0))
    nyu2_test = list((row.split(',') for row in (data['data/nyu2_test.csv']).decode("utf-8").split('\n') if len(row) > 0))

    shape_rgb = (batch_size, 480, 640, 3)
    shape_depth = (batch_size, 240, 320, 1)

    # Helpful for testing...
    if False:
        nyu2_train = nyu2_train[:10]
        nyu2_test = nyu2_test[:10]

    return data, nyu2_train, nyu2_test, shape_rgb, shape_depth

def get_nyu_train_test_data(batch_size):
    data, nyu2_train, nyu2_test, shape_rgb, shape_depth = get_nyu_data(batch_size)

    train_generator = NYU_BasicAugmentRGBSequence(data, nyu2_train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
    test_generator = NYU_BasicRGBSequence(data, nyu2_test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)

    return train_generator, test_generator

def get_void_data(batch_size, void_data_path, use_void_1500=False):
    if use_void_1500:
        void_train_rgb = list(line.strip() for line in open(void_data_path+'/void_1500/train_image.txt'))
        void_train_depth = list(line.strip() for line in open(void_data_path+'/void_1500/train_ground_truth.txt'))
    else:
        void_train_rgb = list(line.strip() for line in open(void_data_path+'/void_150/train_image.txt'))
        void_train_depth = list(line.strip() for line in open(void_data_path+'/void_150/train_ground_truth.txt'))
    #void_train = [[void_train_rgb[i], void_train_depth[i]] for i in range(0, len(void_train_rgb))]
    void_train = []
    for i in range(len(void_train_rgb)):
        if os.path.isfile(os.path.join(void_data_path, void_train_rgb[i]).replace('image', 'sparse_depth')):
            void_train.append([void_train_rgb[i], void_train_depth[i]])
    shape_rgb = (batch_size, 480, 640, 3)
    shape_depth = (batch_size, 240, 320, 1)
    return void_train[:-1350], void_train[-1350:], shape_rgb, shape_depth

def get_void_train_test_data(batch_size, void_data_path='/home/mikkel/data/void_release', mode='normal', channels=4, dont_interpolate=False, use_void_1500=False):
    void_train, void_test, shape_rgb, shape_depth = get_void_data(batch_size, void_data_path, use_void_1500=use_void_1500)
    print('train set size:', len(void_train), ', val set size:', len(void_test))
    if mode == 'normal':
        train_generator = VOID_BasicAugmentRGBSequence(void_data_path, void_train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
        test_generator = VOID_BasicRGBSequence(void_data_path, void_test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
    elif mode == 'two-branch':
        train_generator = VOID_BasicAugmentRGBDSequence(void_data_path, void_train, batch_size=batch_size, shape_depth=shape_depth, channels=channels, dont_interpolate=dont_interpolate)
        test_generator = VOID_BasicRGBDSequence(void_data_path, void_test, batch_size=batch_size, shape_depth=shape_depth, channels=channels, dont_interpolate=dont_interpolate)
    elif mode == '4channel':
        train_generator = VOID_BasicAugmentRGBDSequence(void_data_path, void_train, batch_size=batch_size, shape_depth=shape_depth, channels=4, dont_interpolate=dont_interpolate)
        test_generator = VOID_BasicRGBDSequence(void_data_path, void_test, batch_size=batch_size, shape_depth=shape_depth, channels=4, dont_interpolate=dont_interpolate)
    elif mode == '5channel':
        train_generator = VOID_BasicAugmentRGBDSequence(void_data_path, void_train, batch_size=batch_size, shape_depth=shape_depth, channels=5, dont_interpolate=dont_interpolate)
        test_generator = VOID_BasicRGBDSequence(void_data_path, void_test, batch_size=batch_size, shape_depth=shape_depth, channels=5, dont_interpolate=dont_interpolate)
    elif mode == 'pred+sparse':
        train_generator = VOID_InitPredSparseSequence(void_data_path, void_train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
        test_generator = VOID_InitPredSparseSequence(void_data_path, void_test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
    return train_generator, test_generator

class VOID_BasicAugmentRGBSequence(Sequence):
    def __init__(self, data_root, data_paths, batch_size, shape_rgb, shape_depth, is_flip=False, is_addnoise=False, is_erase=False):
        self.data_root = data_root
        self.dataset = data_paths
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2, 
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.minDepth = settings.MIN_DEPTH*settings.DEPTH_SCALE #cm
        self.maxDepth = settings.MAX_DEPTH*settings.DEPTH_SCALE #cm

        from sklearn.utils import shuffle
        self.dataset = shuffle(self.dataset, random_state=0)

        self.N = len(self.dataset)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )

        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = np.clip(np.asarray(Image.open( self.data_root+"/"+sample[0] )).reshape(480,640,3)/255,0,1)
            y = np.asarray(np.asarray(Image.open( self.data_root+"/"+sample[1] ))/256.0)
            y = np.clip(y.reshape(480,640,1)*settings.DEPTH_SCALE, self.minDepth, self.maxDepth) # fill missing pixels and convert to cm
            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = nyu_resize(x, 480)
            batch_y[i] = nyu_resize(y, 240)

            if is_apply_policy: batch_x[i], batch_y[i] = self.policy(batch_x[i], batch_y[i])

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

class VOID_BasicRGBSequence(Sequence):
    def __init__(self, data_root, data_paths, batch_size,shape_rgb, shape_depth):
        self.data_root = data_root
        self.dataset = data_paths
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.minDepth = settings.MIN_DEPTH*settings.DEPTH_SCALE #cm
        self.maxDepth = settings.MAX_DEPTH*settings.DEPTH_SCALE #cm

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )
        for i in range(self.batch_size):            
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = np.clip(np.asarray(Image.open( self.data_root+"/"+sample[0] )).reshape(480,640,3)/255,0,1)
            y = np.asarray(np.asarray(Image.open( self.data_root+"/"+sample[1] ))/256.0)
            y = np.clip(y.reshape(480,640,1)*settings.DEPTH_SCALE, self.minDepth, self.maxDepth) # fill missing pixels and convert to cm
            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = nyu_resize(x, 480)
            batch_y[i] = nyu_resize(y, 240)

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

class VOID_BasicAugmentRGBDSequence(Sequence):
    def __init__(self, data_root, data_paths, batch_size, shape_depth, channels=5, is_flip=False, is_addnoise=False, is_erase=False, dont_interpolate=False):
        self.data_root = data_root
        self.dataset = data_paths
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2, 
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.channels = channels
        self.shape_rgbd = (batch_size, 480, 640, self.channels)
        self.shape_depth = shape_depth
        self.minDepth = settings.MIN_DEPTH*settings.DEPTH_SCALE #cm
        self.maxDepth = settings.MAX_DEPTH*settings.DEPTH_SCALE #cm
        self.dont_interpolate = dont_interpolate

        from sklearn.utils import shuffle
        self.dataset = shuffle(self.dataset, random_state=0)

        self.N = len(self.dataset)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_x, batch_y = np.zeros( self.shape_rgbd ), np.zeros( self.shape_depth )

        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = np.clip(np.asarray(Image.open( self.data_root+"/"+sample[0] )).reshape(480,640,3)/255,0,1)
            #iz = DepthNorm(np.clip(np.asarray(Image.open( os.path.join(self.data_root, sample[0]).replace('image', 'interp_depth') ))/256.0*100,10.0,1000.0), maxDepth=self.maxDepth)
            if self.dont_interpolate:
                iz = DepthNorm(np.clip(np.asarray(Image.open( os.path.join(self.data_root, sample[0]).replace('image', 'sparse_depth') ))/256.0*settings.DEPTH_SCALE, self.minDepth, self.maxDepth), maxDepth=self.maxDepth)
            else:
                iz = DepthNorm(np.clip(np.asarray(Image.open( os.path.join(self.data_root, sample[0]).replace('image', 'interp_depth') ))/256.0*settings.DEPTH_SCALE, self.minDepth, self.maxDepth), maxDepth=self.maxDepth)
            vm = None
            if self.channels == 5:
                vm = np.array(Image.open(os.path.join(self.data_root, sample[0]).replace('image', 'validity_map')), dtype=np.float32)
                assert(np.all(np.unique(vm) == [0, 256]))
                vm[vm > 0] = 1

            y = np.asarray(np.asarray(Image.open( self.data_root+"/"+sample[1] ))/256.0)
            y = np.clip(y.reshape(480,640,1)*settings.DEPTH_SCALE, self.minDepth, self.maxDepth) # fill missing pixels and convert to cm
            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_y[i] = nyu_resize(y, 240)

            if is_apply_policy: x, batch_y[i], iz, vm = self.policy(x, batch_y[i], iz=iz, vm=vm)

            if self.channels == 5: 
                batch_x[i] = np.stack([x[:,:,0], x[:,:,1], x[:,:,2], iz, vm], axis=-1)
            else: 
                batch_x[i] = np.stack([x[:,:,0], x[:,:,1], x[:,:,2], iz], axis=-1)

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

class VOID_BasicRGBDSequence(Sequence):
    def __init__(self, data_root, data_paths, batch_size, shape_depth, channels=5, dont_interpolate=False):
        self.data_root = data_root
        self.dataset = data_paths
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.channels = channels
        self.shape_rgbd = (batch_size, 480, 640, self.channels)
        self.shape_depth = shape_depth
        self.minDepth = settings.MIN_DEPTH*settings.DEPTH_SCALE #cm
        self.maxDepth = settings.MAX_DEPTH*settings.DEPTH_SCALE #cm
        self.dont_interpolate = dont_interpolate

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros( self.shape_rgbd ), np.zeros( self.shape_depth )
        for i in range(self.batch_size):            
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = np.clip(np.asarray(Image.open( self.data_root+"/"+sample[0] )).reshape(480,640,3)/255,0,1)

            #iz = DepthNorm(np.clip(np.asarray(Image.open( os.path.join(self.data_root, sample[0]).replace('image', 'interp_depth') ))/256.0*100,10.0,1000.0), maxDepth=self.maxDepth)
            if self.dont_interpolate:
                iz = DepthNorm(np.clip(np.asarray(Image.open( os.path.join(self.data_root, sample[0]).replace('image', 'sparse_depth') ))/256.0*settings.DEPTH_SCALE, self.minDepth, self.maxDepth), maxDepth=self.maxDepth)
            else:
                iz = DepthNorm(np.clip(np.asarray(Image.open( os.path.join(self.data_root, sample[0]).replace('image', 'interp_depth') ))/256.0*settings.DEPTH_SCALE, self.minDepth, self.maxDepth), maxDepth=self.maxDepth)
            vm = None
            y = np.asarray(np.asarray(Image.open( self.data_root+"/"+sample[1] ))/256.0)
            y = np.clip(y.reshape(480,640,1)*settings.DEPTH_SCALE, self.minDepth, self.maxDepth) # fill missing pixels and convert to cm
            y = DepthNorm(y, maxDepth=self.maxDepth)
            if self.channels == 5:
                vm = np.array(Image.open(os.path.join(self.data_root, sample[0]).replace('image', 'validity_map')), dtype=np.float32)
                assert(np.all(np.unique(vm) == [0, 256]))
                vm[vm > 0] = 1
                batch_x[i] = np.stack([x[:,:,0], x[:,:,1], x[:,:,2], iz, vm], axis=-1)
            else:
                batch_x[i] = np.stack([x[:,:,0], x[:,:,1], x[:,:,2], iz], axis=-1)
            batch_y[i] = nyu_resize(y, 240)

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

    def get_sample_image(self, idx):
        batch_x, batch_y = np.zeros( self.shape_rgbd ), np.zeros( self.shape_depth )
        samples = [ ('void_1500/data/classroom6/image/1552696011.5523.png', 'void_1500/data/classroom6/ground_truth_orig/1552696011.5523.png'),\
        ('void_1500/data/mechanical_lab3/image/1552096515.0227.png', 'void_1500/data/mechanical_lab3/ground_truth_orig/1552096515.0227.png'),\
        ('void_1500/data/stairs4/image/1552695287.0660.png', 'void_1500/data/stairs4/ground_truth_orig/1552695287.0660.png'),\
        ('void_1500/data/office3/image/1552625426.4194.png', 'void_1500/data/office3/ground_truth_orig/1552625426.4194.png'),\
        ('void_1500/data/desktop2/image/1552625303.1627.png', 'void_1500/data/desktop2/ground_truth_orig/1552625303.1627.png'),\
        ('void_1500/data/plants4/image/1552695221.0711.png', 'void_1500/data/plants4/ground_truth_orig/1552695221.0711.png') ]
        
        i = 0
        index = min(idx, 5)
        sample = samples[index]
        x = np.clip(np.asarray(Image.open( self.data_root+"/"+sample[0] )).reshape(480,640,3)/255,0,1)

        #iz = DepthNorm(np.clip(np.asarray(Image.open( os.path.join(self.data_root, sample[0]).replace('image', 'interp_depth') ))/256.0*100,10.0,1000.0), maxDepth=self.maxDepth)
        if self.dont_interpolate:
            iz = DepthNorm(np.clip(np.asarray(Image.open( os.path.join(self.data_root, sample[0]).replace('image', 'sparse_depth') ))/256.0*settings.DEPTH_SCALE, self.minDepth, self.maxDepth), maxDepth=self.maxDepth)
        else:
            iz = DepthNorm(np.clip(np.asarray(Image.open( os.path.join(self.data_root, sample[0]).replace('image', 'interp_depth') ))/256.0*settings.DEPTH_SCALE, self.minDepth, self.maxDepth), maxDepth=self.maxDepth)
        vm = None
        y = np.asarray(np.asarray(Image.open( self.data_root+"/"+sample[1] ))/256.0)
        y = np.clip(y.reshape(480,640,1)*settings.DEPTH_SCALE, self.minDepth, self.maxDepth) # fill missing pixels and convert to cm
        y = DepthNorm(y, maxDepth=self.maxDepth)
        if self.channels == 5:
            vm = np.array(Image.open(os.path.join(self.data_root, sample[0]).replace('image', 'validity_map')), dtype=np.float32)
            assert(np.all(np.unique(vm) == [0, 256]))
            vm[vm > 0] = 1
            batch_x[i] = np.stack([x[:,:,0], x[:,:,1], x[:,:,2], iz, vm], axis=-1)
        else:
            batch_x[i] = np.stack([x[:,:,0], x[:,:,1], x[:,:,2], iz], axis=-1)
        batch_y[i] = nyu_resize(y, 240)

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

class VOID_ImuAidedRGBSequence(Sequence):
    def __init__(self, data_root, data_paths, batch_size,shape_rgb, shape_depth, use_void_1500=False):
        self.data_root = data_root
        self.dataset = data_paths
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.shape_rgb = shape_rgb
        self.shape_sz = (batch_size, 480, 640, 2)
        self.shape_depth = (batch_size, 480, 640, 1)#shape_depth
        self.minDepth = settings.MIN_DEPTH*settings.DEPTH_SCALE #cm
        self.maxDepth = settings.MAX_DEPTH*settings.DEPTH_SCALE #cm
        from sklearn.utils import shuffle
        self.dataset = shuffle(self.dataset, random_state=0)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x, batch_sz, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_sz ), np.zeros( self.shape_depth )
        for i in range(self.batch_size):            
            index = min((idx * self.batch_size) + i, self.N-1)
            sample = self.dataset[index]
            im = np.clip(np.asarray(Image.open( self.data_root+"/"+sample[0] ))/255,0,1).reshape(480,640,3)
            #iz = np.clip(np.asarray(Image.open( os.path.join(self.data_root, sample[0]).replace('image', 'interp_depth') ))/256.0/10.0,0,1).reshape(480,640)
            iz = DepthNorm(np.clip(np.asarray(Image.open( os.path.join(self.data_root, sample[0]).replace('image', 'interp_depth') ))/256.0*settings.DEPTH_SCALE,self.minDepth,self.maxDepth).reshape(480,640,1), maxDepth=self.maxDepth)
            vm = np.array(Image.open(os.path.join(self.data_root, sample[0]).replace('image', 'validity_map')), dtype=np.float32).reshape(480,640,1)
            assert(np.all(np.unique(vm) == [0, 256]))
            vm[vm > 0] = 1
            gt = np.asarray(np.asarray(Image.open( self.data_root+"/"+sample[1] ))/256.0)
            #y[y <= 0] = 0.0
            #v = y.astype(np.float32)
            #v[y > 0] = 1.0
            #v[y > 10] = 0.0
            #y = np.clip(interpolate_depth(y, v).reshape(480,640,1)*100, 10.0, 1000.0) # fill missing pixels and convert to cm
            gt = np.clip(gt.reshape(480,640,1)*settings.DEPTH_SCALE, self.minDepth, self.maxDepth) # fill missing pixels and convert to cm
            gt = DepthNorm(gt, maxDepth=self.maxDepth)
            batch_x[i] = nyu_resize(im, 480)            
            batch_sz[i] = np.stack([iz, vm], axis=-1).reshape(480,640,2)
            #batch_x[i] = np.stack([im[:,:,0], im[:,:,1], im[:,:,2], iz, vm], axis=-1).reshape(480,640,5)
            batch_y[i] = nyu_resize(gt, 480)

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return [batch_x, batch_sz], batch_y

class VOID_InitPredSparseSequence(Sequence):
    def __init__(self, data_root, data_paths, batch_size,shape_rgb, shape_depth, use_void_1500=False):
        self.data_root = data_root
        self.dataset = data_paths
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.minDepth = settings.MIN_DEPTH*settings.DEPTH_SCALE #cm
        self.maxDepth = settings.MAX_DEPTH*settings.DEPTH_SCALE #cm
        from sklearn.utils import shuffle
        self.dataset = shuffle(self.dataset, random_state=0)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x1, batch_x2, batch_y = np.zeros( self.shape_depth ), np.zeros( self.shape_depth ), np.zeros( self.shape_depth )
        for i in range(self.batch_size):            
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x1 = DepthNorm(np.clip(np.asarray(Image.open( os.path.join(self.data_root, sample[0]).replace('image', 'prediction') ))/256.0*settings.DEPTH_SCALE,self.minDepth,self.maxDepth).reshape(480,640,1), maxDepth=self.maxDepth)
            x2 = DepthNorm(np.clip(np.asarray(Image.open( os.path.join(self.data_root, sample[0]).replace('image', 'interp_depth') ))/256.0*settings.DEPTH_SCALE,self.minDepth,self.maxDepth).reshape(480,640,1), maxDepth=self.maxDepth)
            y = DepthNorm(np.clip(np.asarray(Image.open( os.path.join(self.data_root, sample[1]) ))/256.0*settings.DEPTH_SCALE,self.minDepth,self.maxDepth).reshape(480,640,1), maxDepth=self.maxDepth)            

            batch_x1[i] = nyu_resize(x1, 240)
            batch_x1[i] = nyu_resize(x2, 240)
            batch_y[i] = nyu_resize(y, 240)
            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()
        return [batch_x1, batch_x2], batch_y

class NYU_BasicAugmentRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, shape_rgb, shape_depth, is_flip=False, is_addnoise=False, is_erase=False):
        self.data = data
        self.dataset = dataset
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2, 
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0

        from sklearn.utils import shuffle
        self.dataset = shuffle(self.dataset, random_state=0)

        self.N = len(self.dataset)

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )

        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[0]]) )).reshape(480,640,3)/255,0,1)
            y = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[1]]) )).reshape(480,640,1)/255*self.maxDepth,0,self.maxDepth)
            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = nyu_resize(x, 480)
            batch_y[i] = nyu_resize(y, 240)

            if is_apply_policy: batch_x[i], batch_y[i] = self.policy(batch_x[i], batch_y[i])

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

class NYU_BasicRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size,shape_rgb, shape_depth):
        self.data = data
        self.dataset = dataset
        self.batch_size = batch_size
        self.N = len(self.dataset)
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )
        for i in range(self.batch_size):            
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]

            x = np.clip(np.asarray(Image.open( BytesIO(self.data[sample[0]]))).reshape(480,640,3)/255,0,1)
            y = np.asarray(Image.open(BytesIO(self.data[sample[1]])), dtype=np.float32).reshape(480,640,1).copy().astype(float) / 10.0
            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = nyu_resize(x, 480)
            batch_y[i] = nyu_resize(y, 240)

            # DEBUG:
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i])/maxDepth,0,1), idx, i)
        #exit()

        return batch_x, batch_y

#================
# Unreal dataset
#================

import cv2
from skimage.transform import resize

def get_unreal_data(batch_size, unreal_data_file='unreal_data.h5'):
    shape_rgb = (batch_size, 480, 640, 3)
    shape_depth = (batch_size, 240, 320, 1)

    # Open data file
    import h5py
    data = h5py.File(unreal_data_file, 'r')

    # Shuffle
    from sklearn.utils import shuffle
    keys = shuffle(list(data['x'].keys()), random_state=0)

    # Split some validation
    unreal_train = keys[:len(keys)-100]
    unreal_test = keys[len(keys)-100:]

    # Helpful for testing...
    if False:
        unreal_train = unreal_train[:10]
        unreal_test = unreal_test[:10]

    return data, unreal_train, unreal_test, shape_rgb, shape_depth

def get_unreal_train_test_data(batch_size):
    data, unreal_train, unreal_test, shape_rgb, shape_depth = get_unreal_data(batch_size)
    
    train_generator = Unreal_BasicAugmentRGBSequence(data, unreal_train, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth)
    test_generator = Unreal_BasicAugmentRGBSequence(data, unreal_test, batch_size=batch_size, shape_rgb=shape_rgb, shape_depth=shape_depth, is_skip_policy=True)

    return train_generator, test_generator

class Unreal_BasicAugmentRGBSequence(Sequence):
    def __init__(self, data, dataset, batch_size, shape_rgb, shape_depth, is_flip=False, is_addnoise=False, is_erase=False, is_skip_policy=False):
        self.data = data
        self.dataset = dataset
        self.policy = BasicPolicy( color_change_ratio=0.50, mirror_ratio=0.50, flip_ratio=0.0 if not is_flip else 0.2, 
                                    add_noise_peak=0 if not is_addnoise else 20, erase_ratio=-1.0 if not is_erase else 0.5)
        self.batch_size = batch_size
        self.shape_rgb = shape_rgb
        self.shape_depth = shape_depth
        self.maxDepth = 1000.0
        self.N = len(self.dataset)
        self.is_skip_policy = is_skip_policy

    def __len__(self):
        return int(np.ceil(self.N / float(self.batch_size)))

    def __getitem__(self, idx, is_apply_policy=True):
        batch_x, batch_y = np.zeros( self.shape_rgb ), np.zeros( self.shape_depth )
        
        # Useful for validation
        if self.is_skip_policy: is_apply_policy=False

        # Augmentation of RGB images
        for i in range(batch_x.shape[0]):
            index = min((idx * self.batch_size) + i, self.N-1)

            sample = self.dataset[index]
            
            rgb_sample = cv2.imdecode(np.asarray(self.data['x/{}'.format(sample)]), 1)
            depth_sample = self.data['y/{}'.format(sample)] 
            depth_sample = resize(depth_sample, (self.shape_depth[1], self.shape_depth[2]), preserve_range=True, mode='reflect', anti_aliasing=True )
            
            x = np.clip(rgb_sample/255, 0, 1)
            y = np.clip(depth_sample, 10, self.maxDepth)
            y = DepthNorm(y, maxDepth=self.maxDepth)

            batch_x[i] = x
            batch_y[i] = y

            if is_apply_policy: batch_x[i], batch_y[i] = self.policy(batch_x[i], batch_y[i])
                
            #self.policy.debug_img(batch_x[i], np.clip(DepthNorm(batch_y[i],self.maxDepth)/self.maxDepth,0,1), index, i)

        return batch_x, batch_y
