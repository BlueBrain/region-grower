"""Utils module"""
import os
import json
from collections import defaultdict
import numpy as np
import pandas as pd


class NumpyEncoder(json.JSONEncoder):
    """To encode numpy arrays"""

    def default(self, o):  # pylint: disable=method-hidden
        """encoder"""
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        return json.JSONEncoder.default(self, o)


def create_morphologies_dict(dat_file, morph_path, ext=".asc"):
    """ Create dict to load the morphologies from a directory, with dat file """
    morph_name = pd.read_csv(dat_file, sep=" ")
    name_dict = defaultdict(list)
    for morph in morph_name.values:
        name_dict[morph[2]].append(os.path.join(morph_path, morph[0] + ext))
    return name_dict
