Plaidbench
**********

Plaidbench measures the performance of the built-in Keras_ application networks,
using PlaidML_ or TensorFlow_ as a backend.

Installation
============

   git clone https://github.com/plaidml/plaidbench.git
   cd plaidbench
   pip install -r requirements.txt

Basic usage
===========

Run a benchmark::

   python plaidbench.py [--plaid|--noplaid] [--train] NETWORK

The network can be ``inception_v3``, ``resnet50``, ``vgg16``, ``vgg19``,
``xception``, or (with Keras 2.0.6 and later) ``mobilenet``.

Plaidbench measures inference by default, but you can measure training with the
``--train`` option. Use ``--plaid`` to run the PlaidML_ backend, or use
``--no-plaid`` for TensorFlow_.

.. _TensorFlow: https://www.tensorflow.org
.. _PlaidML: https://github.com/vertexai/plaidml
.. _Keras: https://keras.io


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. toctree::
   :maxdepth: 2
   :caption: Contents

