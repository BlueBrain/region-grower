'''App module that defines the command line interface'''
import json
import os
import logging

import click
from tqdm import tqdm

from tns import extract_input
from region_grower.utils import NumpyEncoder

L = logging.getLogger(__name__)


@click.group()
def cli():
    '''A tool for space synthesis management'''


@cli.command(short_help='Create the TMD distribution file')
@click.argument('input_folder', type=str, required=True)
@click.option('-d', '--distribution-filename',
              type=click.Path(file_okay=True, dir_okay=False),
              default='tmd_distributions.json')
@click.option('-p', '--parameter-filename',
              type=click.Path(file_okay=True, dir_okay=False),
              default='tmd_parameters.json')
def train_tmd(input_folder, distribution_filename, parameter_filename):
    '''Generate two JSON files containing the TMD distributions and parmaeter for
    each mtype in INPUT_FOLDER'''
    L.info('Extracting TMD distributions for each mtype.\n'
           'This can take a while...')

    metadata = {'cortical_thickness': [165, 149, 353, 190, 525, 700]}

    distributions = {
        mtype: extract_input.distributions(os.path.join(input_folder, mtype))
        for mtype in tqdm(os.listdir(input_folder))
    }

    tmd_results = {'mtypes': distributions,
                   'metadata': metadata}

    with open(distribution_filename, 'w') as f:
        json.dump(tmd_results, f, cls=NumpyEncoder)

    L.info('Extracting TMD parameters for each mtype.\n '
           'This can take a while...')

    parameters = {
        mtype: extract_input.parameters(os.path.join(input_folder, mtype))
        for mtype in tqdm(os.listdir(input_folder))
    }

    with open(parameter_filename, 'w') as f:
        json.dump(parameters, f, cls=NumpyEncoder, indent=4)

    L.info('Done !')
