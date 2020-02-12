import shutil
import os
from tempfile import TemporaryDirectory
from pathlib import Path

from nose.tools import assert_equal
from click.testing import CliRunner

from region_grower.cli import cli

DATA = Path(__file__).parent / 'data'


def test_cli():
    with TemporaryDirectory('test-report') as folder:
        folder = Path(folder)
        runner = CliRunner()
        result = runner.invoke(cli, ['train-tmd',
                                     str(DATA / 'input-cells'),
                                     '-df', str(folder / 'distributions.json'),
                                     '-pf', str(folder / 'parameters.json')])

        assert_equal(result.exit_code, 0, result.exception)


        result = runner.invoke(cli, ['train-tmd',
                                     str(DATA / 'input-cells'),
                                     '-dc', str(DATA / 'diametrizer_config.json'),
                                     '-df', str(folder / 'distributions_external_diametrizer.json'),
                                     '-pf', str(folder / 'parameters_external_diametrizer.json')
        ])

        assert_equal(result.exit_code, 0, result.exception)
