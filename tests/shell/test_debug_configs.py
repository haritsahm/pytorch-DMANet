import pytest

from tests.helpers.run_command import run_command


@pytest.mark.slow
def test_debug_default():
    command = ['train.py', 'debug=default', 'data_dir=data/cityscape_fo_segmentation']
    run_command(command)


def test_debug_limit_batches():
    command = ['train.py', 'debug=limit_batches', 'data_dir=data/cityscape_fo_segmentation']
    run_command(command)


def test_debug_overfit():
    command = ['train.py', 'debug=overfit', 'data_dir=data/cityscape_fo_segmentation']
    run_command(command)


@pytest.mark.slow
def test_debug_profiler():
    command = ['train.py', 'debug=profiler', 'data_dir=data/cityscape_fo_segmentation']
    run_command(command)


def test_debug_step():
    command = ['train.py', 'debug=step', 'data_dir=data/cityscape_fo_segmentation']
    run_command(command)


def test_debug_test_only():
    command = ['train.py', 'debug=test_only', 'data_dir=data/cityscape_fo_segmentation']
    run_command(command)
