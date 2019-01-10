'''Utils module'''
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    '''To encode numpy arrays'''
    def default(self, o):  # pylint: disable=method-hidden
        '''encoder'''
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)
