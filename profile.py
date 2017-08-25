#!/usr/bin/env python
from __future__ import print_function

import argparse
import json
import numpy as np
import os
import sys
import time


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
    parser.add_argument('--plaid')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('-v', '--verbose', type=int, nargs='?', const=3)
    parser.add_argument('--result', default='/tmp/result.json')
    parser.add_argument('--callgrind', action='store_true')
    parser.add_argument('module')
    args, remain = parser.parse_known_args()
    print(args, remain)

    if args.plaid:
        import plaidml.keras
        if args.verbose:
            plaidml._internal_set_vlog(args.verbose)
        plaidml.keras.install_backend()
        import plaidml.keras.backend
        with open(args.plaid, 'r') as file_:
            plaidml.keras.backend.set_config(file_.read())
        if args.fp16:
            from keras.backend.common import set_floatx
            set_floatx('float16')

    stop_watch = StopWatch(args.callgrind)
    output = Output()
    globals = {
        '__name__': '__main__',
        'stop_watch': stop_watch,
        'output': output,
    }
    data = {
        'example': args.module
    }
    stop_watch.start_outer()
    try:
        sys.argc = len(remain) + 1
        sys.argv[1:] = remain
        print(sys.argv)
        this_dir = os.path.dirname(os.path.abspath(__file__))
        module = os.path.join(this_dir, 'networks', '%s.py' % args.module)
        execfile(module, globals)
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
