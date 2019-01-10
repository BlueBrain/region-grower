'''App module that defines the command line interface'''
import json
import os

import click
from tqdm import tqdm

import tns
from region_grower.utils import NumpyEncoder


@click.group()
def cli():
    '''A tool for space synthesis management'''


@cli.command(short_help='Create the TMD distribution file')
@click.argument('input_folder', type=str, required=True)
@click.argument('output_filename', type=str, required=True)
def train_tmd(input_folder, output_filename):
    '''Generate a JSON file in OUTPUT_FOLDER containing the TMD distributions for
    each mtype in INPUT_FOLDER'''
    print('Extracting TMD distributions for each mtype.'
          'This can take a while...')

    distributions = {
        mtype: tns.extract_input.distributions(os.path.join(input_folder, mtype))
        for mtype in tqdm(os.listdir(input_folder))
    }

    output_folder = '.'
    with open(os.path.join(output_folder, output_filename), 'w') as f:
        json.dump(distributions, f, cls=NumpyEncoder)

    print('Done !')
