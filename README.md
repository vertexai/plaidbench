# Keras application network benchmark
Measure performance of the built-in Keras application networks Inception-V3,
ResNet50, VGG19, and Xception.

To install:

`pip install -r requirements.txt`

If you want to use the PlaidML backend, install it like so:

`pip install plaidml-keras`

Otherwise, you can also use the TensorFlow backend:

`pip install tensorflow`

To run a benchmark on a network:

`python profile.py [--plaid] {inception_v3,resnet50,vgg19,xception}`


