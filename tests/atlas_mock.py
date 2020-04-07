import json
import tempfile
from os.path import join

import numpy as np

from voxcell import VoxelData


def save_to_directory(directory_path, voxel_size,
                      brain_regions_volume,
                      hierarchy_dict, **files):
    voxel_data = VoxelData(brain_regions_volume, (voxel_size, ) * 3)
    voxel_data.save_nrrd(join(directory_path, 'brain_regions.nrrd'))
    with open(join(directory_path, 'hierarchy.json'), 'w') as jf:
        json.dump(hierarchy_dict, jf)
    for filename, volume in files.items():
        voxel_data.with_data(volume).save_nrrd(join(directory_path, filename))
    return


class TemporaryAtlasDirectory:

    def __init__(self, voxel_size, brain_regions_volume, hierarchy_dict, **files):
        self.nrrd_files = files
        self.brain_regions_volume = brain_regions_volume
        self.hierarchy_dict = hierarchy_dict
        self.voxel_size = voxel_size
        return

    def __enter__(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.tempdirname = self.tempdir.__enter__()
        return self.tempdirname

    def __exit__(self, exception_type, value, traceback):
        return self.tempdir.__exit__(exception_type, value, traceback)


def small_O1(folder_path):
    '''Dump a small O1 atlas in folder path'''
    # flip so bottom is [..., 0, ...]
    brain_regions = np.stack([np.flip(
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 2, 2, 2, 2, 2, 2, 2, 2, 0],
                  [0, 3, 3, 3, 3, 3, 3, 3, 3, 0],
                  [0, 4, 4, 4, 4, 4, 4, 4, 4, 0],
                  [0, 5, 5, 5, 5, 5, 5, 5, 5, 0],
                  [0, 5, 5, 5, 5, 5, 5, 5, 5, 0],
                  [0, 6, 6, 6, 6, 6, 6, 6, 6, 0],
                  [0, 6, 6, 6, 6, 6, 6, 6, 6, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), axis=0)] * 10)

    hierarchy = {'id': 0,
                 'name': 'root',
                 'acronym': 'root',
                 'children': [
                     {'id': 1,
                      'name': 'Layer 1',
                      'acronym': 'L1'},
                     {'id': 2,
                      'name': 'Layer 2',
                      'acronym': 'L2'},
                     {'id': 3,
                      'name': 'Layer 3',
                      'acronym': 'L3'},
                     {'id': 4,
                      'name': 'Layer 4',
                      'acronym': 'L4'},
                     {'id': 5,
                      'name': 'Layer 5',
                      'acronym': 'L5'},
                     {'id': 6,
                      'name': 'Layer 6',
                      'acronym': 'L6'}]}

    # orientation is identity quaternion
    orientation = np.full(brain_regions.shape + (4, ), np.nan)
    orientation[brain_regions > 0, :4] = [1, 0, 0, 0]

    PHy = np.full(brain_regions.shape, np.nan)
    boundaries_y = [1000, 800, 700, 600, 500, 300, 0]

    # set depths only inside the brain region
    y = np.linspace(boundaries_y[0], boundaries_y[-1], PHy.shape[1])
    for y_index in range(PHy.shape[1] - 2):
        PHy[:, y_index + 1, :] = y[y_index]

    placement_hints = {f'[PH]{layer}' + '.nrrd':
                       np.stack([np.float32(brain_regions > 0) * boundaries_y[n]
                                 for n in (layer, layer-1)], axis=-1)
                       for layer in range(1, 7)}

    nrrd_files = dict(**{'orientation.nrrd': orientation,
                         '[PH]y.nrrd': PHy},
                      **placement_hints)

    voxel_size = (boundaries_y[0] - boundaries_y[-1]) // len(boundaries_y)
    save_to_directory(folder_path, voxel_size, brain_regions, hierarchy, **nrrd_files)
