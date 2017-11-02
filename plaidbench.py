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

from six import exec_

import argparse
import errno
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


def has_plaid():
    try:
        import plaidml.keras
        return True
    except ImportError:
        return False


def value_check(examples, epochs, batch_size):
    if epochs > examples:
        raise ValueError('The number of epochs must be less than the number of examples.')
    if batch_size > (examples // epochs):
        raise ValueError('The number of examples per epoch must be greater than the batch size.')
    if examples % epochs != 0:
        raise ValueError('The number of examples must be divisible by the number of epochs.')
    if (examples // epochs) % batch_size != 0:
        raise ValueError('The number of examples per epoch is not divisble by the batch size.')


def train(x_train, y_train, epoch_size, model, batch_size, compile_stop_watch, epochs, stop_watch, output):
    # Training
    x = x_train[:epoch_size]
    y = y_train[:epoch_size]
    model.train_on_batch(x_train[0:batch_size], y_train[0:batch_size])
    
    compile_stop_watch.stop()

    for i in range(epochs):
        if i == 1:
            print('Doing the main timing')
        stop_watch.start()
        history = model.fit(x=x, y=y, batch_size=batch_size, epochs=1, shuffle=False, initial_epoch=0)
        stop_watch.stop()
        time.sleep(.025 * random.random())
        if i == 0:
            output.contents = [history.history['loss']]
    output.contents = np.array(output.contents)


def inference(network, model, batch_size, compile_stop_watch, output, x_train, examples, stop_watch):
    # Inference
    y = model.predict(x=x_train, batch_size=batch_size)
    
    compile_stop_watch.stop()
    output.contents = y
    print('Warmup')

    for i in range(32//batch_size + 1):
        y = model.predict(x=x_train, batch_size=batch_size)

    # Now start the clock and run 100 batches
    print('Doing the main timing')

    for i in range(examples//batch_size):
        stop_watch.start()
        y = model.predict(x=x_train, batch_size=batch_size)
        stop_watch.stop()
        time.sleep(.025 * random.random())


def setup(train, epoch_size, batch_size):
    # Setup
    if train:
        # Training setup
        from keras.datasets import cifar10
        from keras.utils.np_utils import to_categorical
        print('Loading the data')
        (x_train, y_train_cats), (x_test, y_test_cats) = cifar10.load_data()
        x_train = x_train[:epoch_size]
        y_train_cats = y_train_cats[:epoch_size]
        y_train = to_categorical(y_train_cats, num_classes=1000)
    else:
        # Inference setup
        this_dir = os.path.dirname(os.path.abspath(__file__))
        cifar_path = os.path.join(this_dir, 'cifar16.npy')
        x_train = np.load(cifar_path).repeat(1 + batch_size//16, axis=0)[:batch_size]
        y_train_cats = None
        y_train = None
    return x_train, y_train


def load_model(module, x_train):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    module = os.path.join(this_dir, 'networks', '%s.py' % module)
    globals = {}
    exec_(open(module).read(), globals)
    x_train = globals['scale_dataset'](x_train)
    model = globals['build_model']()
    print("Model loaded.")
    return module, x_train, model


def run_intial(batch_size, compile_stop_watch, network, model):
    print("Compiling and running initial batch, batch_size={}".format(batch_size))
    compile_stop_watch.start()
    optimizer = 'sgd'
    if network[:3] == 'vgg':
        from keras.optimizers import SGD
        optimizer = SGD(lr=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])


SUPPORTED_NETWORKS = ['inception_v3', 'mobilenet', 'resnet50', 'vgg16', 'vgg19', 'xception']

def main():
    exit_status = 0
    parser = argparse.ArgumentParser()
    plaidargs = parser.add_mutually_exclusive_group()
    plaidargs.add_argument('--plaid', action='store_true')
    plaidargs.add_argument('--no-plaid', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('-v', '--verbose', type=int, nargs='?', const=3)
    parser.add_argument('--result', default='/tmp/plaidbench_results')
    parser.add_argument('--callgrind', action='store_true')
    parser.add_argument('-n', '--examples', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--blanket-run', action='store_true')
    parser.add_argument('--print-stacktraces', action='store_true')
    parser.add_argument('module', choices=SUPPORTED_NETWORKS)
    args = parser.parse_args()

    # Plaid, fp16, and verbosity setup
    if args.plaid or (not args.no_plaid and has_plaid()):
        print('Using PlaidML backend.')
        import plaidml.keras
        if args.verbose:
            plaidml._internal_set_vlog(args.verbose)
        plaidml.keras.install_backend()
    if args.fp16:
        from keras.backend.common import set_floatx
        set_floatx('float16')

    # variable declaration/intialization
    batch_size = int(args.batch_size)
    epochs = args.epochs
    examples = args.examples
    epoch_size = examples // epochs
    networks = []
    output = Output()

    # Stopwatch and Output intialization
    stop_watch = StopWatch(args.callgrind)
    compile_stop_watch = StopWatch(args.callgrind)
    
    # Blanket run - runs every supported network
    if args.blanket_run:
        data = {}
        outputs = {}
        networks = list(SUPPORTED_NETWORKS)

        print("Plaid Blanket Run: setting examples size to 256 for speed")
        examples = 256;

        if args.plaid or (not args.no_plaid and has_plaid()):
            import plaidml
            data['plaid'] = plaidml.__version__
        else:
            data['plaid'] = None

        data['train'] = args.train
        data['blanket_run'] = True
        outputs['run_configuration'] = data.copy()
    else:
        networks.append(args.module)

    
    for network in networks:
        print("\nCurrent network being run : " + network)  
        args.module = network
        network_data = {}

        # Run network
        try:
            # Setup
            x_train, y_train = setup(args.train, epoch_size, batch_size)
        
            # Start stopwatches
            stop_watch.start_outer()
            compile_stop_watch.start_outer()

            # Loading the model
            module, x_train, model = load_model(args.module, x_train)

            # Prep the model and run an initial un-timed batch
            run_intial(batch_size, compile_stop_watch, args.module, model)

            # training run
            if args.train:
                value_check(examples, epochs, batch_size)
                train(x_train, y_train, epoch_size, model, batch_size, compile_stop_watch, epochs, stop_watch, output)
             inference run
            else:
                inference(args.module, model, batch_size, compile_stop_watch, output, x_train, examples, stop_watch)

            # Stop stopwatches
            stop_watch.stop()
            compile_stop_watch.stop()

            # Record stopwatch times
            execution_duration = stop_watch.elapsed()
            compile_duration = compile_stop_watch.elapsed()
            
            # Record data
            network_data['execution_duration'] = execution_duration
            network_data['compile_duration'] = compile_duration
            network_data['precision'] = output.precision

            # Print statement
            print('Example finished, elapsed: {} (compile), {} (execution)'.format(compile_duration, execution_duration))

        # Error handling
        except Exception as ex:
            # Print statements
            print(ex)
            print('Set --print-stacktraces to see the entire traceback')

            # Record error
            network_data['exception'] = str(ex)
            
            # Set new exist status
            exit_status = -1

            # stacktrace loop
            if args.print_stacktraces:
                raise NotImplementedError                
        
        # stores network data in dictionary
        if args.blanket_run:
            outputs[network] = network_data

        # write all data to result.json / report.npy if single run
        else:
            network_data['example'] = network
            try:
                os.makedirs(args.result)
            except OSError as ex:
                if ex.errno != errno.EEXIST:
                    print(ex)
                    return
            with open(os.path.join(args.result, 'result.json'), 'w') as out:
                json.dump(network_data, out)
            if isinstance(output.contents, np.ndarray):
                np.save(os.path.join(args.result, 'result.npy'), output.contents)
            # close
            sys.exit(exit_status)

    # write all data to report.json if blanket run
    if args.blanket_run:
        try:
            os.makedirs(args.result)
        except OSError as ex:
            if ex.errno != errno.EEXIST:
                print(ex)
                return
        with open(os.path.join(args.result, 'report.json'), 'w') as out:
            json.dump(outputs, out)

    # close
    sys.exit(exit_status)


if __name__ == '__main__':
    main()
