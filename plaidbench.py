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
import datetime
import math
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
from asq.initiators import query


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

    def clear(self):
        self.__total = 0.0


class Output(object):
    def __init__(self):
        self.contents = None
        self.precision = 'untested'


def printf(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def getColor(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = -1
    g = -1
    b = -1

    if (0 <= h_i and h_i < 1):
        r = v 
        g = t 
        b = p 
    elif (1 <= h_i and h_i < 2):
        r = q
        g = v 
        b = p 
    elif (2 <= h_i and h_i < 3):
        r = p 
        g = v 
        b = t 
    elif (3 <= h_i and h_i < 4):
        r = p 
        g = q 
        b = v 
    elif (4 <= h_i and h_i < 5):
        r = t 
        g = p 
        b = v 
    else:
        r = v 
        g = p 
        b = q 

    import math
    r = int(r * 256)
    g = int(g * 256)
    b = int(b * 256)

    color = 'rgb(' + str(r) + ', ' + str(g) + ', ' + str(b) + ')'
    color = '#%02x%02x%02x' % (r, g, b)

    return color


def has_plaid():
    try:
        import plaidml.keras
        return True
    except ImportError:
        return False


def value_check(examples, epochs, batch_size):
    if epochs >= examples:
        raise ValueError('The number of epochs must be less than the number of examples.')
    if batch_size >= (examples // epochs):
        raise ValueError('The number of examples per epoch must be greater than the batch size.')
    if examples % epochs != 0:
        raise ValueError('The number of examples must be divisible by the number of epochs.')
    if examples % batch_size != 0:
        raise ValueError('The number of examples must be divisible by the batch size.')
    if (examples // epochs) % batch_size != 0:
        raise ValueError('The number of examples per epoch is not divisible by the batch size.')


def train(x_train, y_train, epoch_size, model, batch_size, compile_stop_watch, 
          epochs, stop_watch, output, network):
    # Training
    stop_watch.clear()
    compile_stop_watch.clear()

    compile_stop_watch.start_outer()
    stop_watch.start_outer()
    print(stop_watch.elapsed())

    run_initial(batch_size, compile_stop_watch, network, model)
    model.train_on_batch(x_train[0:batch_size], y_train[0:batch_size])

    compile_stop_watch.stop()

    x = x_train[:epoch_size]
    y = y_train[:epoch_size]
    
    for i in range(epochs):
        if i == 1:
            printf('Doing the main timing')
        stop_watch.start()
        history = model.fit(
            x=x, y=y, batch_size=batch_size, epochs=1, shuffle=False, initial_epoch=0)
        stop_watch.stop()
        time.sleep(.025 * random.random())
        if i == 0:
            output.contents = [history.history['loss']]
    output.contents = np.array(output.contents)
    stop_watch.stop()


def inference(network, model, batch_size, compile_stop_watch, output, x_train, 
              examples, stop_watch):
    # Inference
    stop_watch.clear()
    compile_stop_watch.clear()

    compile_stop_watch.start_outer()
    stop_watch.start_outer()
    
    run_initial(batch_size, compile_stop_watch, network, model);
    y = model.predict(x=x_train, batch_size=batch_size)
    
    compile_stop_watch.stop()
    
    output.contents = y

    for i in range(32 // batch_size + 1):
        y = model.predict(x=x_train, batch_size=batch_size)

    for i in range(examples // batch_size):
        stop_watch.start()
        y = model.predict(x=x_train, batch_size=batch_size)
        stop_watch.stop()

    stop_watch.stop()


def setup(train, epoch_size, batch_size):
    # Setup
    if train:
        # Training setup
        from keras.datasets import cifar10
        from keras.utils.np_utils import to_categorical
        printf('Loading the data')
        (x_train, y_train_cats), (x_test, y_test_cats) = cifar10.load_data()
        x_train = x_train[:epoch_size]
        y_train_cats = y_train_cats[:epoch_size]
        y_train = to_categorical(y_train_cats, num_classes=1000)
    else:
        # Inference setup
        this_dir = os.path.dirname(os.path.abspath(__file__))
        cifar_path = os.path.join(this_dir, 'cifar16.npy')
        x_train = np.load(cifar_path).repeat(1 + batch_size // 16, axis=0)[:batch_size]
        y_train_cats = None
        y_train = None
    return x_train, y_train


def load_model(module, x_train):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    module = os.path.join(this_dir, 'networks', '%s.py' % module)
    globals = {}
    exec_(open(module).read(), globals)
    x_train = globals['scale_dataset'](x_train)
    
    # when run, this function'sw sub process can access gpu id
    model = globals['build_model']()
    
    printf("Model loaded.")
    return module, x_train, model


def run_initial(batch_size, compile_stop_watch, network, model):
    print("Compiling and running initial batch, batch_size={}".format(batch_size))
    optimizer = 'sgd'
    if network[:3] == 'vgg':
        from keras.optimizers import SGD
        optimizer = SGD(lr=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])


def plot_v3(data, column, column_order, ymax):
    g = sns.FacetGrid(
        data,
        col=column,
        col_order = column_order,
        sharex=False,
        size = 6,
        aspect = .33
    )

    g.map(
        sns.barplot,
        "model", "time per example (seconds)", "batch",
        hue_order = list(set(data['batch'])).sort(),
        order = list(set(data['batch'])).sort()
    )

    axes = np.array(g.axes.flat)
    #hue_start = random.random()
    for ax in axes:
        #ax.hlines(.0003, -0.5, 0.5, linestyle='--', linewidth=1, color=getColor(hue_start, .6, .9))
        ax.set_ylim(0, ymax)

    if ymax == 0:
        print('isZero')
        ymax = 1
        #plt.yticks(np.arange(0, ymax + (ymax * .1), ymax/10))
    else:
        plt.yticks(np.arange(0, ymax + (ymax * .1), ymax/10))

    return plt.gcf(), axes


def set_labels(fig, axes, labels, batch_list, model_count):
    for i, ax in enumerate(axes):
        increment = .75 / len(batch_list)
        illusory = []
        
        if len(batch_list) % 2 == 0:        
            foo = increment / 2
            bar = -1 * foo
            illusory.append(foo)
            illusory.append(bar)

            for j in range((len(batch_list) - 2) / 2):
                foo = foo + increment
                bar = -1 * foo
                illusory.append(foo)
                illusory.append(bar)    
        else:
            illusory.append(0)
            half_len = (len(batch_list) - 1) / 2

            for j in range(half_len):
                illusory.append((increment + (increment * j)))
                illusory.append(-1 * (increment + (increment * j)))
                
        illusory.sort()
        ax.set_xticks(illusory) 
        batch_list.sort()
        ax.set_xticklabels(batch_list)

        ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        #ax.grid(b=True, which='minor')
        ax.grid(b=True, which='both', linewidth=.6)
        
        ax.set_xlabel(labels[i])
        ax.set_ylabel("")
        ax.set_title("")
    axes.flat[0].set_ylabel("Time (sec)")
    
    for x in range(model_count):
        sns.despine(ax=axes[x], left=True)
    
    fig.suptitle("Single example runtime\nby batch size", verticalalignment='top', fontsize=11, y='.99', horizontalalignment='center')
    plt.subplots_adjust(top=0.91)
    

def set_style():
    sns.set_style("whitegrid", {    
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


def color_bars(axes, colors, networks, batches):
    for i in range(networks/batches):
        for x in range(len(axes[i].patches)):
            illusory = axes[i].patches[x]
            illusory.set_color(colors[(i * batches) + x])
            illusory.set_edgecolor('black')
            if len(axes[i].patches) == 1:
                illusory.set_hatch('//') 
                illusory.set_color('grey')
                illusory.set_edgecolor('black')


def date_converter(obj):
    if isinstance(obj, datetime.datetime):
        return obj.__str__()


def save_run_info(uber_list, title_str):
    title_str = title_str + ''
    with open(title_str, 'w') as outfile:
        json.dump(uber_list, outfile, default=date_converter)


def generate_plot(df, title_str):
    set_style()
    col_order = (list(set(df['model'])))
    max_time = (float(max(df['time per example (seconds)'])))

    exponent = np.floor(np.log10(np.abs(max_time))).astype(int)
    base_10 = 10
    if exponent > 0:
        base_10 = 1
    else:
        for number in range(1, np.abs(exponent)):
            base_10 = base_10 * 10
    max_time = ((math.ceil(base_10 * max_time)) / base_10)

    palette = []
    palette_dict = {}

    gradient_step = .99 / (len(list(set(df['batch']))))

    num = -1
    golden_ratio = 0.618033988749895
    h = random.random()

    for x in df['model']:
        if x not in palette_dict:
            palette_dict[x] = h
            h += golden_ratio
            h = h % 1
    
    for x in palette_dict:
        num = palette_dict[x]
        gradient = gradient_step
        for y in list(set(df['batch'])):
            color = getColor(num, gradient, 1 - gradient)
            palette.append(color)
            gradient = gradient + gradient_step

    fig, axes = plot_v3(df, "model", col_order, max_time)
    labels = (list(set(df['model'])))
    set_labels(fig, axes, labels, list(set(df['batch'])), len(labels))
    color_bars(axes, palette, len(df['model']), len(list(set(df['batch']))))

    title = ''
    if title_str != '':
        title = title_str + '.png'
    else:
        title = time.strftime("plaidbench %Y-%m-%d-%H:%M.png")
    print("\nsaving figure '" + title + "'")
    fig.savefig(title)


# Original networks
SUPPORTED_NETWORKS = ['inception_v3', 'mobilenet', 'resnet50', 'vgg16', 'vgg19', 'xception']

def main():
    exit_status = 0
    parser = argparse.ArgumentParser()
    plaidargs = parser.add_mutually_exclusive_group()
    plaidargs.add_argument('--plaid', action='store_true')
    plaidargs.add_argument('--no-plaid', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('--result', default='/tmp/plaidbench_results')
    parser.add_argument('--callgrind', action='store_true')
    parser.add_argument('-n', '--examples', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--blanket-run', action='store_true')
    parser.add_argument('--print-stacktraces', action='store_true')
    parser.add_argument('-s', '--save', type=str, default=None)
    parser.add_argument('--regraph', type=str, default=None)
    parser.add_argument('--graph', action='store_true')
    args1 = parser.parse_known_args()
    if args1[0].blanket_run == False:
        parser.add_argument('module', choices=SUPPORTED_NETWORKS)
    args = parser.parse_args()

    if args.regraph != None:
        uber_dict = {}
        with open(args.regraph) as saved_file:
            for line in saved_file:
                uber_dict = json.loads(line)
        d = {}
        d['model'] = uber_dict['model']
        d['time per example (seconds)'] = uber_dict['time per example (seconds)']
        d['batch'] = uber_dict['batch']
        d['name'] = uber_dict['name']
        machine_info = uber_dict['machine_info']
        df = pd.DataFrame.from_dict(d)
        generate_plot(df, args.regraph)
        sys.exit(exit_status)

    # Plaid, fp16, and verbosity setup
    if args.plaid or (not args.no_plaid and has_plaid()):
        printf('Using PlaidML backend.')
        import plaidml.keras
        if args.verbose:
            plaidml._internal_set_vlog(args.verbose)
        plaidml.keras.install_backend()
    if args.fp16:
        from keras.backend.common import set_floatx
        set_floatx('float16')

    examples = -1
    if args.examples == None:
        if args.blanket_run:
            examples = 256
        else:
            examples = 1024
    else:
        examples = args.examples
    batch_size = int(args.batch_size)
    epochs = args.epochs
    epoch_size = examples // epochs
    networks = []
    batch_list = []
    output = Output()

    # Stopwatch and Output intialization
    stop_watch = StopWatch(args.callgrind)
    compile_stop_watch = StopWatch(args.callgrind)
    
    # Blanket run - runs every supported network
    if args.blanket_run:
        data = {}
        outputs = {}
        networks = list(SUPPORTED_NETWORKS)
        batch_list = [1, 4, 8, 16]

        if args.plaid or (not args.no_plaid and has_plaid()):
            import plaidml
            data['plaid'] = plaidml.__version__
        else:
            data['plaid'] = None

        data['example_size'] = examples
        data['train'] = args.train
        data['blanket_run'] = True
        outputs['run_configuration'] = data.copy()
    else:
        networks.append(args.module)
        batch_list.append(args.batch_size)

    for network in networks:
        printf("\nCurrent network being run : " + network)  
        args.module = network
        network_data = {}

        for batch in batch_list:
            batch_size = batch
            printf('Running {0} examples with {1}, batch size {2}'.format(examples, network, batch))
        
            # Run network w/ batch_size
            try:
                value_check(examples, epochs, batch_size)
                
                # Setup
                x_train, y_train = setup(args.train, epoch_size, batch_size)

                # Loading the model
                module, x_train, model = load_model(args.module, x_train)
                
                if args.train:
                    # training run
                    train(x_train, y_train, epoch_size, model, batch_size, 
                          compile_stop_watch, epochs, stop_watch, output, network)
                else:
                    # inference run
                    inference(args.module, model, batch_size, compile_stop_watch, 
                              output, x_train, examples, stop_watch)

                # Record stopwatch times
                execution_duration = stop_watch.elapsed()
                compile_duration = compile_stop_watch.elapsed()

                network_data['compile_duration'] = compile_duration
                network_data['execution_duration'] = execution_duration / examples
                network_data['precision'] = output.precision
                network_data['example_size'] = examples
                network_data['batch_size'] = batch_size
                network_data['model'] = network

                # Print statement
                printf('Example finished, elapsed: {} (compile), {} (execution)'.format(
                    compile_duration, execution_duration))

            # Error handling
            except Exception as ex:
                # Print statements
                printf(ex)
                printf('Set --print-stacktraces to see the entire traceback')

                # Record error
                network_data['exception'] = str(ex)
                
                # Set new exist status
                exit_status = -1

                # stacktrace loop
                if args.print_stacktraces:
                    raise              

            # stores network data in dictionary
            if args.blanket_run:
                composite_str = network + " : " + str(batch_size)
                outputs[composite_str] = dict(network_data)
            # write all data to result.json / report.npy if single run
            else:
                try:
                    os.makedirs(args.result)
                except OSError as ex:
                    if ex.errno != errno.EEXIST:
                        printf(ex)
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
                printf(ex)
                return
        with open(os.path.join(args.result, 'report.json'), 'w') as out:
            json.dump(outputs, out)

        # attempting to get info about users env
        from plaidml import plaidml_setup
        import platform

        userSys = platform.uname()
        userPyV = platform.python_version()
        machine_info = []
        for info in userSys:
            machine_info.append(info)
        machine_info.append(userPyV)

        # creating dict with completed runs
        d = outputs
        runs = {}
        for key, values in d.items():
            if 'compile_duration' in values:
                runs[key] = values

        models_list = []
        executions_list = []
        batch_list2 = []
        name = []
        uber_list = pd.DataFrame()

        for x, y in sorted(runs.items()):
            models_list.append(y['model'])
            executions_list.append( y['execution_duration'] / examples )
            batch_list2.append(y['batch_size'])
            name.append(y['model'] + " : " + str(y['batch_size']))
        
        uber_list['model'] = models_list
        uber_list['time per example (seconds)'] = executions_list
        uber_list['batch'] = batch_list2
        uber_list['name'] = name
        
        ctx = plaidml.Context()
        devices, _ = plaidml.devices(ctx, limit=100, return_all=True)
        for dev in devices:      
            plt.suptitle(str(dev))
            machine_info.append(str(dev))

        if args.graph:
            generate_plot(uber_list, args.save)
        
        if args.save != None:
            machine_info.append(datetime.datetime.now())
            uber_list = uber_list.to_dict()
            uber_list['machine_info'] = machine_info
            print("saving run data as '" + args.save + "'")   
            save_run_info(uber_list, args.save)
        
    # close
    sys.exit(exit_status)


if __name__ == '__main__':
    main()
