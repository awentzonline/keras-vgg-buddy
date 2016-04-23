import argparse

import numpy as np
import six
from keras_vgg_buddy import get_model, IMAGENET_CLASSES, load_and_preprocess_image


parser = argparse.ArgumentParser(description='Classify an image with VGG16.')
parser.add_argument('img_path', type=six.text_type)
parser.add_argument('--weights', type=six.text_type, default='vgg16_weights.h5')
args = parser.parse_args()

img = load_and_preprocess_image(args.img_path, width=224, square=True)
img_channels, img_rows, img_cols = img.shape
vgg = get_model(img_rows, img_cols, weights_path=args.weights)
vgg.compile(optimizer='adam', loss='mse')
classes = vgg.predict(img[None, ...])
best_class = np.argmax(classes)
print('Best guess: {}'.format(IMAGENET_CLASSES[best_class]))
