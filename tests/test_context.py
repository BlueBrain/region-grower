import mock
import os
import numpy as np

from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_raises, assert_equal, assert_almost_equal

from voxcell.nexus.voxelbrain import Atlas
from region_grower.context import SpaceContext

_path = os.path.dirname(__file__)

from mock import MagicMock

def test_get_layer():
    context = SpaceContext(MagicMock(),
                           os.path.join(_path, 'test_distributions.json'),
                           os.path.join(_path, 'test_parameters.json'))

    cumulative_thickness = [-100, -10, 30, 200]
    context.depths.lookup = MagicMock(return_value = 56)

    # The first 'argument' position, is actually irrelevent
    # as we are mocking directly its corresponding depth
    assert_equal(context._get_layer(None, cumulative_thickness),
                 3)

    context.depths.lookup = MagicMock(return_value = -56)
    assert_equal(context._get_layer(None, cumulative_thickness),
                 1)

    # Should the 2 following cases raise instead ?
    context.depths.lookup = MagicMock(return_value = -200)
    assert_equal(context._get_layer(None, cumulative_thickness),
                 0)

    context.depths.lookup = MagicMock(return_value = 1000)
    assert_equal(context._get_layer(None, cumulative_thickness),
                 4)


def test_context():
    np.random.seed(0)

    atlas = MagicMock()
    context = SpaceContext(atlas,
                           os.path.join(_path, 'test_distributions.json'),
                           os.path.join(_path, 'test_parameters.json'))
    context._get_orientation = MagicMock(return_value = [-0.33126975, -0.91100982, -0.2456043 ])
    context._cumulative_thicknesses = MagicMock(return_value = [ 125, 225, 450, 575, 900, 1250])
    context._get_layer = MagicMock(return_value = 1)

    neuron = context.synthesize([6001.39477031, 772.6250185, 3778.3698741], 'L2_TPC:A')

    # This tests that input orientations are not mutated by the synthesize() call
    assert_array_almost_equal(context.tmd_parameters['L2_TPC:A']['apical']['orientation'],
                              [[0.0, 1.0, 0.0]])

    assert_array_almost_equal(neuron.soma.points,
                              np.array([[-9.12136  ,  0.34562492 , 0.        ],
                                        [-3.028861 , -8.329533   , 0.        ],
                                        [ 6.060588 , -4.2906003  , 0.        ],
                                        [ 8.941994 ,  1.9075108  , 0.        ],
                                        [ 6.8666873,  6.0370903  , 0.        ],
                                        [ 0.1265565,  9.05724    , 0.        ],
                                        [-4.1508913,  7.8628845  , 0.        ]],
                                       dtype=np.float32))

    assert_equal(len(neuron.root_sections), 8)
    assert_array_almost_equal(neuron.root_sections[0].points,
                              np.array([[ 0.12655652,  9.05724 ,    1.2442882 ],
                                        [ 0.1398302 , 10.007194,    1.3747933 ],
                                        [ 0.13791543, 10.887132,    1.6561068 ]],
                                       dtype=np.float32))


    assert_array_almost_equal(neuron.root_sections[0].diameters,
                              np.array([0.6, 0.6, 0.6 ], dtype=np.float32))
