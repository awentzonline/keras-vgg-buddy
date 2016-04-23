Keras VGG buddy
===============
Lends a hand in adding a trained VGG16 network to your [Keras](http://keras.io/) model.

Pre-trained model
-----------------
[Here's a download](
https://github.com/awentzonline/keras-vgg-buddy/releases/download/0.0.1/vgg16_weights.h5)
which contains only the convolutional layers of VGG16, cutting it down to 10% of the full size.
If you want to use the fully-connected layers download the full set of parameters from:
[original source of weights](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3).

This code was originally adapted from the Keras neural style example. The weights
are available under the [CC BY-NC 4.0](http://creativecommons.org/licenses/by-nc/4.0/)
license.
