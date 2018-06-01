# Copyright 2018 Vertex.AI

import logging
import re
import subprocess
import sys

import pytest
import pytest_virtualenv

log = logging.getLogger(__name__)


@pytest.fixture(scope='function')
def benchenv(pytestconfig):
    with pytest_virtualenv.VirtualEnv() as venv:
        venv.run([venv.python, 'setup.py', 'develop'], capture=False, cd=pytestconfig.rootdir)
        if sys.platform == 'win32':
            venv.installer = [venv.virtualenv / 'Scripts' / 'pip.exe', 'install']
            venv.plaidbench = venv.virtualenv / 'Scripts' / 'plaidbench.exe'
        else:
            venv.installer = [venv.virtualenv / 'bin' / 'pip', 'install']
            venv.plaidbench = venv.virtualenv / 'bin' / 'plaidbench'
        yield venv


class TestSetup(object):

    def test_initial_setup(self, benchenv):
        benchenv.run([benchenv.plaidbench])

    def install_deps(self, benchenv, excinfo):
        matches = re.search(r': ([^:\n]+)\n', excinfo.value.output)
        assert matches is not None
        pkgs = matches.group(1).split()
        log.info('Missing packages: {}'.format(' '.join(pkgs)))
        benchenv.run(benchenv.installer + pkgs, capture=True)

    @pytest.mark.skip(reason="Currently in development")
    @pytest.mark.parametrize('args', [
        ['--examples=1', 'onnx', 'squeezenet'],
        ['--examples=1', 'onnx', '--tensorflow', 'squeezenet'],
        ['--examples=1', 'keras', 'mobilenet'],
        ['--examples=1', 'keras', '--tensorflow', 'mobilenet'],
    ])
    def test_install(self, benchenv, args):
        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            benchenv.run([benchenv.plaidbench] + args, capture=True)
        self.install_deps(benchenv, excinfo)
        benchenv.run([benchenv.plaidbench] + args, capture=True)
