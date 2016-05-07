import os

import h5py
import numpy as np
from keras import backend as K
from keras.layers import (AveragePooling2D, Convolution2D, Dense, Dropout,
    Flatten, Input, MaxPooling2D, ZeroPadding2D)
from keras.models import Model


def img_from_vgg(x):
    '''Decondition an image from the VGG16 model.'''
    x = x.transpose((1, 2, 0))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:,:,::-1]  # to RGB
    return x


def img_to_vgg(x):
    '''Condition an image for use with the VGG16 model.'''
    x = x[:,:,::-1]  # to BGR
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68
    x = x.transpose((2, 0, 1))
    return x


class VGG16(object):
    def __init__(self, num_rows, num_cols, weights_path='vgg16_weights.h5',
            pool_mode='avg', last_layer='conv5_1', learning_phase=None):
        self.learning_phase = learning_phase
        self.last_layer = last_layer
        self.net = get_model(num_rows, num_cols, weights_path=weights_path,
            pool_mode=pool_mode, last_layer=last_layer)
        self.net_input = self.net.get_layer('vgg_input')
        self._f_layer_outputs = {}

    def get_f_layer(self, layer_name):
        '''Create a function for the response of a layer.'''
        inputs = [self.net_input]
        if self.learning_phase is not None:
            inputs.append(K.learning_phase())
        return K.function(inputs, [self.get_layer_output(layer_name)])

    def get_layer_output(self, name):
        '''Get symbolic output of a layer.'''
        if not name in self._f_layer_outputs:
            layer = self.net.get_layer(name)
            self._f_layer_outputs[name] = layer.output
        return self._f_layer_outputs[name]

    def get_layer_output_shape(self, name):
        layer = self.net.get_layer(name)
        return layer.output_shape

    def get_features(self, x, layers):
        '''Evaluate layer outputs for `x`'''
        if not layers:
            return None
        inputs = [self.net.input]
        if self.learning_phase is not None:
            inputs.append(self.learning_phase)
        f = K.function(inputs, [self.get_layer_output(layer_name) for layer_name in layers])
        feature_outputs = f([x])
        features = dict(zip(layers, feature_outputs))
        return features


def get_model(num_rows, num_cols, weights_path='vgg16_weights.h5', pool_mode='max',
        trainable=False, last_layer=None):
    vgg_input = Input(shape=(3, num_rows, num_cols), name='vgg_input')
    vgg_stack = add_vgg_to_layer(vgg_input, pool_mode=pool_mode, trainable=trainable,
        weights_path=weights_path, last_layer=last_layer)
    model = Model(input=vgg_input, output=vgg_stack)
    return model


def add_vgg_to_layer(input_layer, trainable=False, pool_mode='max',
        weights_path='vgg16_weights.h5', last_layer=None):
    layers = get_layers(pool_mode=pool_mode, last_layer=last_layer,
        trainable=trainable, weights_path=weights_path)
    # load the weights of the VGG16 networks
    assert os.path.exists(weights_path), 'Model weights not found (see "--vgg-weights" parameter).'
    f = h5py.File(weights_path)
    last_layer = input_layer
    for k, layer in zip(range(f.attrs['nb_layers']), layers):
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        if isinstance(layer, Convolution2D) and K._BACKEND == 'theano':
            weights[0] = np.array(weights[0])[:, :, ::-1, ::-1]
        last_layer = layer(last_layer)
        layer.trainable = trainable
        layer.set_weights(weights)
    f.close()
    return last_layer


def get_layers(pool_mode='max', last_layer=None, weights_path='vgg16_weights.h5',
        trainable=False):
    assert pool_mode in ('avg', 'max'), '`pool_mode` must be "avg" or "max"'
    if pool_mode == 'avg':
        pool_class = AveragePooling2D
    else:
        pool_class = MaxPooling2D
    layers = [
        ZeroPadding2D((1, 1), name='vgg_input_zp'),
        Convolution2D(64, 3, 3, activation='relu', name='conv1_1'),
        ZeroPadding2D((1, 1), name='conv1_1_zp'),
        Convolution2D(64, 3, 3, activation='relu', name='conv1_2'),
        pool_class((2, 2), strides=(2, 2), name='pool1'),

        ZeroPadding2D((1, 1), name='pool1_zp'),
        Convolution2D(128, 3, 3, activation='relu', name='conv2_1'),
        ZeroPadding2D((1, 1), name='conv2_1_zp'),
        Convolution2D(128, 3, 3, activation='relu', name='conv2_2'),
        pool_class((2, 2), strides=(2, 2), name='pool2'),

        ZeroPadding2D((1, 1), name='pool2_zp'),
        Convolution2D(256, 3, 3, activation='relu', name='conv3_1'),
        ZeroPadding2D((1, 1), name='conv3_1_zp'),
        Convolution2D(256, 3, 3, activation='relu', name='conv3_2'),
        ZeroPadding2D((1, 1), name='conv3_2_zp'),
        Convolution2D(256, 3, 3, activation='relu', name='conv3_3'),
        pool_class((2, 2), strides=(2, 2), name='pool3'),

        ZeroPadding2D((1, 1), name='pool3_zp'),
        Convolution2D(512, 3, 3, activation='relu', name='conv4_1'),
        ZeroPadding2D((1, 1), name='conv4_1_zp'),
        Convolution2D(512, 3, 3, activation='relu', name='conv4_2'),
        ZeroPadding2D((1, 1), name='conv4_2_zp'),
        Convolution2D(512, 3, 3, activation='relu', name='conv4_3'),
        pool_class((2, 2), strides=(2, 2), name='pool4'),

        ZeroPadding2D((1, 1), name='pool4_zp'),
        Convolution2D(512, 3, 3, activation='relu', name='conv5_1'),
        ZeroPadding2D((1, 1), name='conv5_1_zp'),
        Convolution2D(512, 3, 3, activation='relu', name='conv5_2'),
        ZeroPadding2D((1, 1), name='conv5_2_zp'),
        Convolution2D(512, 3, 3, activation='relu', name='conv5_3'),
        pool_class((2, 2), strides=(2, 2), name='pool5'),

        Flatten(),
        Dense(4096, activation='relu', name='fc6'),
        Dropout(0.5, name='fc6do'),
        Dense(4096, activation='relu', name='fc7'),
        Dropout(0.5, name='fc7do'),
        Dense(1000, activation='softmax', name='vgg_classes'),
    ]
    if last_layer is not None:
        while layers[-1].name != last_layer:
            layers.pop()
    return layers
