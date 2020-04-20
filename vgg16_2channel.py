#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.applications.vgg19 import VGG19
from keras.models import Model

vgg19 = VGG19(weights='imagenet')
vgg19.summary() # To check which layers will be omitted in 'pretrained' model

# Load part of the VGG without the top layers into 'pretrained' model
pretrained = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_pool').output)
pretrained.summary()

#%% Prepare model template with 4 input channels
config = pretrained.get_config() # run config['layers'][i] for reference
                                 # to restore layer-by layer structure

from keras.layers import Input, Conv2D, MaxPooling2D
from keras import optimizers

# For training from scratch change kernel_initializer to e.g.'VarianceScaling'
inputs = Input(shape=(224, 224, 4), name='input_17')
# block 1
x = Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='zeros', name='block1_conv1')(inputs)
x = Conv2D(64, (3,3), padding='same', activation='relu', kernel_initializer='zeros', name='block1_conv2')(x)
x = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(x)

# block 2
x = Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer='zeros', name='block2_conv1')(x)
x = Conv2D(128, (3,3), padding='same', activation='relu', kernel_initializer='zeros', name='block2_conv2')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), name='block2_pool')(x)

# block 3
x = Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer='zeros', name='block3_conv1')(x)
x = Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer='zeros', name='block3_conv2')(x)
x = Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer='zeros', name='block3_conv3')(x)
x = Conv2D(256, (3,3), padding='same', activation='relu', kernel_initializer='zeros', name='block3_conv4')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), name='block3_pool')(x)

# block 4
x = Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer='zeros', name='block4_conv1')(x)
x = Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer='zeros', name='block4_conv2')(x)
x = Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer='zeros', name='block4_conv3')(x)
x = Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer='zeros', name='block4_conv4')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), name='block4_pool')(x)

# block 5
x = Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer='zeros', name='block5_conv1')(x)
x = Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer='zeros', name='block5_conv2')(x)
x = Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer='zeros', name='block5_conv3')(x)
x = Conv2D(512, (3,3), padding='same', activation='relu', kernel_initializer='zeros', name='block5_conv4')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2,2), name='block5_pool')(x)

vgg_template = Model(inputs=inputs, outputs=x)

vgg_template.compile(optimizer=optimizers.RMSprop(lr=2e-4),
                     loss='categorical_crossentropy',
                     metrics=['acc'])


#%% Rewrite the weight loading/modification function
import numpy as np

layers_to_modify = ['block1_conv1'] # Turns out the only layer that changes
                                    # shape due to 4th channel is the first
                                    # convolution layer.

for layer in pretrained.layers: # pretrained Model and template have the same
                                # layers, so it doesn't matter which to 
                                # iterate over.

    if layer.get_weights() != []: # Skip input, pooling and no weights layers

        target_layer = vgg_template.get_layer(name=layer.name)

        if layer.name in layers_to_modify:

            kernels = layer.get_weights()[0]
            biases  = layer.get_weights()[1]

            kernels_extra_channel = np.concatenate((kernels,
                                                    kernels[:,:,-1:,:]),
                                                    axis=-2) # For channels_last

            target_layer.set_weights([kernels_extra_channel, biases])

        else:
            target_layer.set_weights(layer.get_weights())


#%% Save 4 channel model populated with weights for futher use    

vgg_template.save('vgg19_modified_clear.hdf5')