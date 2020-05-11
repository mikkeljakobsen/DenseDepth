import sys

from keras import applications
from keras.models import Model, load_model
from keras.layers import Input, InputLayer, Conv2D, Activation, LeakyReLU, Concatenate, Lambda
from layers import BilinearUpSampling2D
from loss import depth_loss_function
from keras.layers.merge import concatenate

def create_model(existing='', is_twohundred=False, is_halffeatures=True, channels=3):
        
    if len(existing) == 0:
        print('Loading base model (DenseNet)..')
        if channels != 3:
            base_model = applications.DenseNet169(input_shape=(None, None, channels), include_top=False, weights=None)
            pretrained = applications.DenseNet169(input_shape=(None, None, 3), include_top=False)
            for layer in pretrained.layers:
                print(layer.name)
                if layer.get_weights() != []:  # Skip input, pooling and no weights layers
                    target_layer = base_model.get_layer(name=layer.name)
                    if layer.name != 'conv1/conv':
                        target_layer.set_weights(layer.get_weights())
        # Encoder Layers
        elif is_twohundred:
            base_model = applications.DenseNet201(input_shape=(None, None, 3), include_top=False)
        else:
            base_model = applications.DenseNet169(input_shape=(480, 640, 3), include_top=False)

        print('Base model loaded.')

        # Starting point for decoder
        base_model_output_shape = base_model.layers[-1].output.shape

        # Layer freezing?
        for layer in base_model.layers: layer.trainable = True

        # Starting number of decoder filters
        if is_halffeatures:
            decode_filters = int(int(base_model_output_shape[-1])/2)
        else:
            decode_filters = int(base_model_output_shape[-1])
        print('Decode filters =', decode_filters)

        # Define upsampling layer
        def upproject(tensor, filters, name, concat_with):
            up_i = BilinearUpSampling2D((2, 2), name=name+'_upsampling2d')(tensor)
            up_i = Concatenate(name=name+'_concat')([up_i, base_model.get_layer(concat_with).output]) # Skip connection
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            return up_i

        # Decoder Layers
        decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape, name='conv2')(base_model.output)

        decoder = upproject(decoder, int(decode_filters/2), 'up1', concat_with='pool3_pool')
        decoder = upproject(decoder, int(decode_filters/4), 'up2', concat_with='pool2_pool')
        decoder = upproject(decoder, int(decode_filters/8), 'up3', concat_with='pool1')
        decoder = upproject(decoder, int(decode_filters/16), 'up4', concat_with='conv1/relu')
        if False: decoder = upproject(decoder, int(decode_filters/32), 'up5', concat_with='input_1')

        # Extract depths (final layer)
        conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(decoder)



        # Create the model
        model = Model(inputs=base_model.input, outputs=conv3)
    else:
        # Load model from file
        if not existing.endswith('.h5'):
            sys.exit('Please provide a correct model file when using [existing] argument.')
        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}
        model = load_model(existing, custom_objects=custom_objects)
        print('\nExisting model loaded.\n')


    print('Model created.')
    
    return model


def create_two_branch_model(existing='', is_twohundred=False, is_halffeatures=True, channels=4):
        

    if len(existing) == 0:
        print('Loading base model (DenseNet)..')

        def crop(dimension, start, end):
            # Crops (or slices) a Tensor on a given dimension from start to end
            # example : to crop tensor x[:, :, 5:10]
            # call slice(2, 5, 10) as you want to crop on the second dimension
            def func(x):
                if dimension == 0:
                    return x[start: end]
                if dimension == 1:
                    return x[:, start: end]
                if dimension == 2:
                    return x[:, :, start: end]
                if dimension == 3:
                    return x[:, :, :, start: end]
                if dimension == 4:
                    return x[:, :, :, :, start: end]
            return Lambda(func)

        input_rgbd = Input(shape=(None, None, channels), name='input_rgbd')
        input_rgb = crop(3, 0, 3)(input_rgbd)
        input_sparse = crop(3, 3, channels)(input_rgbd)
        #base_model_sz_input = Conv2D(3, (3,3), padding='same')(input_sparse)

        # Encoder Layers
        if is_twohundred:
            base_model = applications.DenseNet201(input_shape=(None, None, 3), include_top=False, weights='imagenet', input_tensor=input_rgb)
            #base_model_sz = applications.DenseNet201(input_shape=(None, None, 3), include_top=False, weights='imagenet', input_tensor=concatenate([input_sparse, input_sparse, input_sparse], axis=-1))
            base_model_sz = applications.DenseNet201(input_shape=(None, None, channels-3), include_top=False, weights=None, input_tensor=input_sparse)
        else:
            base_model = applications.DenseNet169(input_shape=(None, None, 3), include_top=False, weights='imagenet', input_tensor=input_rgb)
            #base_model_sz = applications.DenseNet169(input_shape=(None, None, 3), include_top=False, weights='imagenet', input_tensor=concatenate([input_sparse, input_sparse, input_sparse], axis=-1))
            base_model_sz = applications.DenseNet121(input_shape=(None, None, channels-3), include_top=False, weights=None, input_tensor=input_sparse)
            pretrained = applications.DenseNet121(input_shape=(None, None, 3), include_top=False, weights='imagenet')
        for layer in pretrained.layers:
            print(layer.name)
            if layer.get_weights() != []:  # Skip input, pooling and no weights layers
                target_layer = base_model_sz.get_layer(name=layer.name)
                if layer.name != 'conv1/conv': # Initialize imagenet weights in all layers except the first conv1 layer where the channels do not match
                    target_layer.set_weights(layer.get_weights())

        print('Base model loaded.')

        # Layer freezing?
        for layer in base_model.layers: 
            layer.trainable = True
        for layer in base_model_sz.layers: 
            layer.trainable = True
            layer.name = layer.name + str("_sz")

        # Starting point for decoder
        encoder_output = concatenate([base_model.output, base_model_sz.output], axis=-1)
        base_model_output_shape = encoder_output.shape

        #base_model_output_shape = base_model.layers[-1].output.shape
        #base_model_output_shape_sz = base_model_sz.layers[-1].output.shape


        # Starting number of decoder filters
        if is_halffeatures:
            decode_filters = int(int(base_model_output_shape[-1])/2)
        else:
            decode_filters = int(base_model_output_shape[-1])

        # Define upsampling layer
        def upproject(tensor, filters, name, concat_with):
            up_i = BilinearUpSampling2D((2, 2), name=name+'_upsampling2d')(tensor)
            up_i = Concatenate(name=name+'_concat')([up_i, base_model.get_layer(concat_with).output, base_model_sz.get_layer(concat_with+str("_sz")).output]) # Skip connection
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            return up_i

        # Decoder Layers
        decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape, name='conv2')(encoder_output)
        decoder = upproject(decoder, int(decode_filters/2), 'up1', concat_with='pool3_pool')
        decoder = upproject(decoder, int(decode_filters/4), 'up2', concat_with='pool2_pool')
        decoder = upproject(decoder, int(decode_filters/8), 'up3', concat_with='pool1')
        decoder = upproject(decoder, int(decode_filters/16), 'up4', concat_with='conv1/relu')
        if False: decoder = upproject(decoder, int(decode_filters/32), 'up5', concat_with='input_1')

        # Extract depths (final layer)
        conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(decoder)

        # Create the model
        model = Model(inputs=input_rgbd, outputs=conv3)
    else:
        # Load model from file
        if not existing.endswith('.h5'):
            sys.exit('Please provide a correct model file when using [existing] argument.')
        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}
        model = load_model(existing, custom_objects=custom_objects, compile=False)
        print('\nExisting model loaded.\n')

    print('Model created.')
    
    return model

def create_two_branch_model_very_late_fusion(existing='', is_twohundred=False, is_halffeatures=True, channels=4):
        
    if len(existing) == 0:
        print('Loading base model (DenseNet)..')

        def crop(dimension, start, end):
            # Crops (or slices) a Tensor on a given dimension from start to end
            # example : to crop tensor x[:, :, 5:10]
            # call slice(2, 5, 10) as you want to crop on the second dimension
            def func(x):
                if dimension == 0:
                    return x[start: end]
                if dimension == 1:
                    return x[:, start: end]
                if dimension == 2:
                    return x[:, :, start: end]
                if dimension == 3:
                    return x[:, :, :, start: end]
                if dimension == 4:
                    return x[:, :, :, :, start: end]
            return Lambda(func)

        input_rgbd = Input(shape=(None, None, channels), name='input_rgbd')
        input_rgb = crop(3, 0, 3)(input_rgbd)
        input_sparse = crop(3, 3, channels)(input_rgbd)
        #base_model_sz_input = Conv2D(3, (3,3), padding='same')(input_sparse)

        # Encoder Layers
        if is_twohundred:
            base_model = applications.DenseNet201(input_shape=(None, None, 3), include_top=False, weights='imagenet', input_tensor=input_rgb)
            #base_model_sz = applications.DenseNet201(input_shape=(None, None, 3), include_top=False, weights='imagenet', input_tensor=concatenate([input_sparse, input_sparse, input_sparse], axis=-1))
            base_model_sz = applications.DenseNet201(input_shape=(None, None, channels-3), include_top=False, weights=None, input_tensor=input_sparse)
        else:
            base_model = applications.DenseNet169(input_shape=(None, None, 3), include_top=False, weights='imagenet', input_tensor=input_rgb)
            #base_model_sz = applications.DenseNet169(input_shape=(None, None, 3), include_top=False, weights='imagenet', input_tensor=concatenate([input_sparse, input_sparse, input_sparse], axis=-1))
            base_model_sz = applications.DenseNet169(input_shape=(None, None, channels-3), include_top=False, weights=None, input_tensor=input_sparse)
            pretrained = applications.DenseNet169(input_shape=(None, None, 3), include_top=False, weights='imagenet')
            #base_model_sz = applications.DenseNet121(input_shape=(None, None, channels-3), include_top=False, weights=None, input_tensor=input_sparse)
            #pretrained = applications.DenseNet121(input_shape=(None, None, 3), include_top=False, weights='imagenet')
        for layer in pretrained.layers:
            print(layer.name)
            if layer.get_weights() != []:  # Skip input, pooling and no weights layers
                target_layer = base_model_sz.get_layer(name=layer.name)
                if layer.name != 'conv1/conv': # Initialize imagenet weights in all layers except the first conv1 layer where the channels do not match
                    target_layer.set_weights(layer.get_weights())

        print('Base model loaded.')

        # Layer freezing?
        for layer in base_model.layers: 
            layer.trainable = True
        for layer in base_model_sz.layers: 
            layer.trainable = True
            layer.name = layer.name + str("_sz")

        # Starting point for decoder
        #encoder_output = concatenate([base_model.output, base_model_sz.output], axis=-1)
        #base_model_output_shape = encoder_output.shape

        base_model_output_shape = base_model.layers[-1].output.shape
        base_model_output_shape_sz = base_model_sz.layers[-1].output.shape


        # Starting number of decoder filters
        decode_filters = int(int(base_model_output_shape[-1])/2)
        decode_filters_sz = int(int(base_model_output_shape_sz[-1])/2)

        # Define upsampling layer
        def upproject(tensor, filters, name, concat_with):
            up_i = BilinearUpSampling2D((2, 2), name=name+'_upsampling2d')(tensor)
            up_i = Concatenate(name=name+'_concat')([up_i, base_model.get_layer(concat_with).output]) # Skip connection
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            return up_i

        # Decoder Layers
        decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape, name='conv2')(base_model.output)
        decoder = upproject(decoder, int(decode_filters/2), 'up1', concat_with='pool3_pool')
        decoder = upproject(decoder, int(decode_filters/4), 'up2', concat_with='pool2_pool')
        decoder = upproject(decoder, int(decode_filters/8), 'up3', concat_with='pool1')
        decoder = upproject(decoder, int(decode_filters/16), 'up4', concat_with='conv1/relu')
        if False: decoder = upproject(decoder, int(decode_filters/32), 'up5', concat_with='input_1')

        def upproject_sz(tensor, filters, name, concat_with):
            up_i = BilinearUpSampling2D((2, 2), name=name+'_upsampling2d')(tensor)
            up_i = Concatenate(name=name+'_concat')([up_i, base_model_sz.get_layer(concat_with+str("_sz")).output]) # Skip connection
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            return up_i

        # Decoder Layers
        decoder_sz = Conv2D(filters=decode_filters_sz, kernel_size=1, padding='same', input_shape=base_model_output_shape, name='conv2_sz')(base_model.output)
        decoder_sz = upproject_sz(decoder_sz, int(decode_filters_sz/2), 'up1_sz', concat_with='pool3_pool')
        decoder_sz = upproject_sz(decoder_sz, int(decode_filters_sz/4), 'up2_sz', concat_with='pool2_pool')
        decoder_sz = upproject_sz(decoder_sz, int(decode_filters_sz/8), 'up3_sz', concat_with='pool1')
        decoder_sz = upproject_sz(decoder_sz, int(decode_filters_sz/16), 'up4_sz', concat_with='conv1/relu')
        if False: decoder_sz = upproject_sz(decoder_sz, int(decode_filters_sz/32), 'up5_sz', concat_with='input_1')

        # Extract depths (final layer)
        conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(concatenate([decoder, decoder_sz], axis=-1))

        # Create the model
        model = Model(inputs=input_rgbd, outputs=conv3)
    else:
        # Load model from file
        if not existing.endswith('.h5'):
            sys.exit('Please provide a correct model file when using [existing] argument.')
        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}
        model = load_model(existing, custom_objects=custom_objects, compile=False)
        print('\nExisting model loaded.\n')

    print('Model created.')
    
    return model


def create_model_early(existing='', is_twohundred=False, is_halffeatures=True, channels=4):
        

    if len(existing) == 0:
        print('Loading base model (DenseNet)..')

        def crop(dimension, start, end):
            # Crops (or slices) a Tensor on a given dimension from start to end
            # example : to crop tensor x[:, :, 5:10]
            # call slice(2, 5, 10) as you want to crop on the second dimension
            def func(x):
                if dimension == 0:
                    return x[start: end]
                if dimension == 1:
                    return x[:, start: end]
                if dimension == 2:
                    return x[:, :, start: end]
                if dimension == 3:
                    return x[:, :, :, start: end]
                if dimension == 4:
                    return x[:, :, :, :, start: end]
            return Lambda(func)

        input_rgbd = Input(shape=(None, None, channels), name='input_rgbd')
        input_rgb = crop(3, 0, 3)(input_rgbd)
        input_rgb = Conv2D(filters=48, kernel_size=3, padding='same', input_shape=input_rgb.shape, name='conv0')(input_rgb)
        input_sparse = crop(3, 3, channels)(input_rgbd)
        input_sparse = Conv2D(filters=16, kernel_size=3, padding='same', input_shape=input_sparse.shape, name='conv0_sz')(input_sparse)

        # Encoder Layers
        if is_twohundred:
            base_model = applications.DenseNet201(input_shape=(None, None, 64), include_top=False, weights=None, input_tensor=concatenate([input_rgb, input_sparse]))
        else:
            base_model = applications.DenseNet169(input_shape=(None, None, 64), include_top=False, weights=None, input_tensor=concatenate([input_rgb, input_sparse]))
            pretrained = applications.DenseNet169(input_shape=(None, None, 3), include_top=False, weights='imagenet')
        for layer in pretrained.layers:
            print(layer.name)
            if layer.get_weights() != []:  # Skip input, pooling and no weights layers
                target_layer = base_model.get_layer(name=layer.name)
                if layer.name != 'conv1/conv': # Initialize imagenet weights in all layers except the first conv1 layer where the channels do not match
                    target_layer.set_weights(layer.get_weights())

        print('Base model loaded.')

                # Starting point for decoder
        base_model_output_shape = base_model.layers[-1].output.shape

        # Layer freezing?
        for layer in base_model.layers: layer.trainable = True

        # Starting number of decoder filters
        if is_halffeatures:
            decode_filters = int(int(base_model_output_shape[-1])/2)
        else:
            decode_filters = int(base_model_output_shape[-1])
        print('Decode filters =', decode_filters)

        # Define upsampling layer
        def upproject(tensor, filters, name, concat_with):
            up_i = BilinearUpSampling2D((2, 2), name=name+'_upsampling2d')(tensor)
            up_i = Concatenate(name=name+'_concat')([up_i, base_model.get_layer(concat_with).output]) # Skip connection
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convA')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            up_i = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=name+'_convB')(up_i)
            up_i = LeakyReLU(alpha=0.2)(up_i)
            return up_i

        # Decoder Layers
        decoder = Conv2D(filters=decode_filters, kernel_size=1, padding='same', input_shape=base_model_output_shape, name='conv2')(base_model.output)

        decoder = upproject(decoder, int(decode_filters/2), 'up1', concat_with='pool3_pool')
        decoder = upproject(decoder, int(decode_filters/4), 'up2', concat_with='pool2_pool')
        decoder = upproject(decoder, int(decode_filters/8), 'up3', concat_with='pool1')
        decoder = upproject(decoder, int(decode_filters/16), 'up4', concat_with='conv1/relu')
        if False: decoder = upproject(decoder, int(decode_filters/32), 'up5', concat_with='input_1')

        # Extract depths (final layer)
        conv3 = Conv2D(filters=1, kernel_size=3, strides=1, padding='same', name='conv3')(decoder)

        # Create the model
        model = Model(inputs=input_rgbd, outputs=conv3)
    else:
        # Load model from file
        if not existing.endswith('.h5'):
            sys.exit('Please provide a correct model file when using [existing] argument.')
        custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}
        model = load_model(existing, custom_objects=custom_objects, compile=False)
        print('\nExisting model loaded.\n')

    print('Model created.')
    
    return model