# Copyright 2018 Vertex.AI

from click.testing import CliRunner
import pytest

import plaidbench.cli


@pytest.mark.parametrize('args', [
    ['--examples=1', 'onnx', 'squeezenet'],
    ['--examples=1', 'keras', 'mobilenet'],
])
def test_plaidbench(args):
    runner = CliRunner()
    result = runner.invoke(plaidbench.cli.plaidbench, args)
    assert result.exit_code == 0
