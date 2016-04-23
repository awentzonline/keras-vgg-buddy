import argparse

import numpy as np
import six
from keras_vgg_buddy import load_and_preprocess_image, VGG16


parser = argparse.ArgumentParser(description='Classify an image with VGG16.')
parser.add_argument('img_path', type=six.text_type)
parser.add_argument('--weights', type=six.text_type, default='vgg16_weights.h5')
args = parser.parse_args()

img = load_and_preprocess_image(args.img_path, width=224, square=True)
img_channels, img_rows, img_cols = img.shape
vgg = VGG16(img_rows, img_cols, weights_path=args.weights)
some_features = vgg.get_features(img[None, ...], ['conv4_1', 'conv4_2'])
for name, features in some_features.items():
    print('Name: {} Shape: {}'.format(name, features.shape))
