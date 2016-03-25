import os

import h5py
import numpy as np
from keras import backend as K
from keras.layers.convolutional import (
    AveragePooling2D, Convolution2D, MaxPooling2D, ZeroPadding2D)
from keras.models import Graph


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
    def __init__(self, num_rows, num_cols, weights_path='vgg16_weights.h5', pool_mode='avg'):
        self.net = get_model(num_rows, num_cols, weights_path=weights_path, pool_mode=pool_mode)
        self.net_input = self.net.get_input()['vgg_input']
        self._f_layer_outputs = {}

    def get_f_layer(self, layer_name):
        '''Create a function for the response of a layer.'''
        return K.function([self.net_input], [self.get_layer_output(layer_name)])

    def get_layer_output(self, name):
        '''Get symbolic output of a layer.'''
        if not name in self._f_layer_outputs:
            layer = self.net.nodes[name]
            self._f_layer_outputs[name] = layer.get_output()
        return self._f_layer_outputs[name]

    def get_layer_output_shape(self, name):
        layer = self.net.nodes[name]
        return layer.output_shape

    def get_features(self, x, layers):
        '''Evaluate layer outputs for `x`'''
        if not layers:
            return None
        f = K.function([self.net_input], [self.get_layer_output(layer_name) for layer_name in layers])
        feature_outputs = f([x])
        features = dict(zip(layers, feature_outputs))
        return features

    def add_node(self, *args, **kwargs):
        self.net.add_node(*args, **kwargs)


def get_model(num_rows, num_cols, weights_path='vgg16_weights.h5', pool_mode='max', trainable=False):
    assert pool_mode in ('avg', 'max'), '`pool_mode` must be "avg" or "max"'
    if pool_mode == 'avg':
        pool_class = AveragePooling2D
    else:
        pool_class = MaxPooling2D
    model = Graph()
    model.add_input('vgg_input', input_shape=(3, num_rows, num_cols))
    add_vgg_to_graph(model, 'vgg_input', pool_mode=pool_mode, trainable=trainable, weights_path=weights_path)
    return model


def add_vgg_to_graph(model, input_name, trainable=False, pool_mode='max', weights_path='vgg16_weights.h5'):
    assert pool_mode in ('avg', 'max'), '`pool_mode` must be "avg" or "max"'
    if pool_mode == 'avg':
        pool_class = AveragePooling2D
    else:
        pool_class = MaxPooling2D
    layers = [
        (ZeroPadding2D((1, 1)), 'vgg_input_zp', input_name),
        (Convolution2D(64, 3, 3, activation='relu'), 'conv1_1', 'vgg_input_zp'),
        (ZeroPadding2D((1, 1)), 'conv1_1_zp', 'conv1_1'),
        (Convolution2D(64, 3, 3, activation='relu'), 'conv1_2', 'conv1_1_zp'),
        (pool_class((2, 2), strides=(2, 2)), 'pool1', 'conv1_2'),

        (ZeroPadding2D((1, 1)), 'pool1_zp', 'pool1'),
        (Convolution2D(128, 3, 3, activation='relu'), 'conv2_1', 'pool1_zp'),
        (ZeroPadding2D((1, 1)), 'conv2_1_zp', 'conv2_1'),
        (Convolution2D(128, 3, 3, activation='relu'), 'conv2_2', 'conv2_1_zp'),
        (pool_class((2, 2), strides=(2, 2)), 'pool2', 'conv2_2'),

        (ZeroPadding2D((1, 1)), 'pool2_zp', 'pool2'),
        (Convolution2D(256, 3, 3, activation='relu'), 'conv3_1', 'pool2_zp'),
        (ZeroPadding2D((1, 1)), 'conv3_1_zp', 'conv3_1'),
        (Convolution2D(256, 3, 3, activation='relu'), 'conv3_2', 'conv3_1_zp'),
        (ZeroPadding2D((1, 1)), 'conv3_2_zp', 'conv3_2'),
        (Convolution2D(256, 3, 3, activation='relu'), 'conv3_3', 'conv3_2_zp'),
        (pool_class((2, 2), strides=(2, 2)), 'pool3', 'conv3_3'),

        (ZeroPadding2D((1, 1)), 'pool3_zp', 'pool3'),
        (Convolution2D(512, 3, 3, activation='relu'), 'conv4_1', 'pool3_zp'),
        (ZeroPadding2D((1, 1)), 'conv4_1_zp', 'conv4_1'),
        (Convolution2D(512, 3, 3, activation='relu'), 'conv4_2', 'conv4_1_zp'),
        (ZeroPadding2D((1, 1)), 'conv4_2_zp', 'conv4_2'),
        (Convolution2D(512, 3, 3, activation='relu'), 'conv4_3', 'conv4_2_zp'),
        (pool_class((2, 2), strides=(2, 2)), 'pool4', 'conv4_3'),

        (ZeroPadding2D((1, 1)), 'pool4_zp', 'pool4'),
        (Convolution2D(512, 3, 3, activation='relu'), 'conv5_1', 'pool4_zp'),
        (ZeroPadding2D((1, 1)), 'conv5_1_zp', 'conv5_1'),
        (Convolution2D(512, 3, 3, activation='relu'), 'conv5_2', 'conv5_1_zp'),
        (ZeroPadding2D((1, 1)), 'conv5_2_zp', 'conv5_2'),
        (Convolution2D(512, 3, 3, activation='relu'), 'conv5_3', 'conv5_2_zp'),
        (pool_class((2, 2), strides=(2, 2)), 'pool5', 'conv5_3'),
    ]
    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)
    assert os.path.exists(weights_path), 'Model weights not found (see "--vgg-weights" parameter).'
    f = h5py.File(weights_path)
    for k, node_args in zip(range(f.attrs['nb_layers']), layers):
        model.add_node(*node_args)
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        layer, _, _ = node_args
        if isinstance(layer, Convolution2D) and K._BACKEND == 'theano':
            weights[0] = np.array(weights[0])[:, :, ::-1, ::-1]
        layer.set_weights(weights)
        layer.trainable = trainable

    f.close()
    return model
