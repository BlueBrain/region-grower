"""Utils module"""
import json
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd

L = logging.getLogger(__name__)


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
        return json.JSONEncoder.default(self, o)  # pragma: no cover


def create_morphologies_dict(dat_file, morph_path, ext=".asc"):
    """ Create dict to load the morphologies from a directory, with dat file """
    morph_name = pd.read_csv(dat_file, sep=" ")
    name_dict = defaultdict(list)
    for morph in morph_name.values:
        name_dict[morph[2]].append(os.path.join(morph_path, morph[0] + ext))
    return name_dict


def formatted_logger(msg: str, **kwargs) -> None:
    """Add a logger entry if the given condition is True and dump kwargs as JSON in this
    entry.

    Args:
        msg: the message to log (must contain a `%s` to dump the JSON)
        kwargs: entries dumped in JSON format
    """
    if L.isEnabledFor(logging.DEBUG):  # Explicit check to avoid json.dump() when possible
        L.debug(
            msg,
            json.dumps(
                kwargs,
                cls=NumpyEncoder,
            ),
        )
