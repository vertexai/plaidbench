#!/usr/bin/env python

# Copyright 2017 Vertex.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import json
import numpy as np
import os
import sys
import time
import random

class StopWatch(object):
    def __init__(self, use_callgrind):
        self.__start = None
        self.__stop = None
        self.__use_callgrind = use_callgrind
        self.__callgrind_active = False
        self.__total = 0.0

    def start_outer(self):
        # Like start(), but does not turn on callgrind.
        self.__start = time.time()

    def start(self):
        self.__start = time.time()
        if self.__use_callgrind:
          os.system('callgrind_control --instr=on %d' % (os.getpid(),))
          self.__callgrind_active = True

    def stop(self):
        if self.__start is not None:
            stop = time.time()
            self.__total += stop - self.__start
            self.__start = None
        if self.__callgrind_active:
            self.__callgrind_active = False
            os.system('callgrind_control --instr=off %d' % (os.getpid(),))

    def elapsed(self):
        return self.__total


class Output(object):
  def __init__(self):
    self.contents = None
    self.precision = 'untested'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plaid', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('-v', '--verbose', type=int, nargs='?', const=3)
    parser.add_argument('--result', default='/tmp/result.json')
    parser.add_argument('--callgrind', action='store_true')
    parser.add_argument('module')
    args, remain = parser.parse_known_args()

    if args.plaid:
        import plaidml.keras
        if args.verbose:
            plaidml._internal_set_vlog(args.verbose)
        plaidml.keras.install_backend()
    if args.fp16:
        from keras.backend.common import set_floatx
        set_floatx('float16')

    # Load the dataset and scrap everything but the training images
    # cifar10 data is too small, but we can upscale
    from keras.datasets import cifar10
    print("Loading the data")
    (x_train, y_train_cats), (x_test, y_test_cats) = cifar10.load_data()
    y_train_cats = None
    x_test = None
    y_test_cats = None
    # Set a batch size and truncate the number of images
    batch_size = 1
    x_train = x_train[:batch_size]

    stop_watch = StopWatch(args.callgrind)
    output = Output()
    data = {
        'example': args.module
    }
    stop_watch.start_outer()
    try:
        sys.argc = len(remain) + 1
        sys.argv[1:] = remain
        this_dir = os.path.dirname(os.path.abspath(__file__))
        module = os.path.join(this_dir, 'networks', '%s.py' % args.module)
        globals = {}
        execfile(module, globals)

        print("Upscaling the data")
        x_train = globals['scale_dataset'](x_train)

        print("Loading the model")
        model = globals['build_model']()

        # Prep the model and run an initial un-timed batch
        print("Compiling")
        model.compile(optimizer='sgd', loss='categorical_crossentropy',
                      metrics=['accuracy'])

        print("Running initial batch")
        y = model.predict(x=x_train, batch_size=batch_size)
        output.contents = y

        print("Warmup")
        for i in range(10):
            stop_watch.start()

        # Now start the clock and run 100 batches
        print("Doing the main timing")
        for i in range(1000):
            stop_watch.start()
            y = model.predict(x=x_train, batch_size=batch_size)
            stop_watch.stop()
            time.sleep(.025 * random.random())

        stop_watch.stop()
        elapsed = stop_watch.elapsed()
        data['elapsed'] = elapsed
        print('Example finished, elapsed: %s' % elapsed)
        data['precision'] = output.precision
    except Exception as ex:
        print(ex)
        data['exception'] = str(ex)
        raise
    finally:
        with open(args.result, 'w') as out:
            json.dump(data, out)
        if isinstance(output.contents, np.ndarray):
            # Horrible hack of filename choice (in expectation of rewrite)
            np_out_filename = "".join(args.result.split(".")[:-1]) + ".npy"
            np.save(np_out_filename, output.contents)


if __name__ == '__main__':
    main()
