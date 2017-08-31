# Keras application network benchmark
Measure performance of the built-in Keras application networks: Inception-V3,
ResNet50, VGG16, VGG19, Xception, and (in Keras 2.0.6 and later) MobileNet.

To install:

`pip install -r requirements.txt`

If you want to use the PlaidML backend, install it like so:

`pip install plaidml-keras`

Otherwise, you can also use the TensorFlow backend:

`pip install tensorflow`

To run a benchmark on a network:

`python profile.py [--plaid] NETWORK`

where NETWORK is one of the names "inception_v3", "resnet50", "vgg16", "vgg19",
"xception", or "mobilenet".

