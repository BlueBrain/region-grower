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
    assert_array_almost_equal(neuron.soma.points,
                              np.array([[-2.8641052 ,  5.425396  ,  0.        ],
                                        [-6.2937384 ,  0.23849252,  0.        ],
                                        [-0.65263325, -2.2691152 ,  0.        ],
                                        [ 4.181802  , -2.9605203 ,  0.        ],
                                        [ 0.08732622,  6.2494955 ,  0.        ],
                                        [ 1.4765941 ,  2.9637198 ,  0.        ],
                                        [-2.089914  , -5.7473774 ,  0.        ]],
                                       dtype=np.float32))

    assert_equal(len(neuron.root_sections), 7)
    assert_array_almost_equal(neuron.root_sections[0].points,
                              np.array([[ 4.181802 , -2.9605203,  3.680869 ],
                                        [ 4.2885404, -3.1940644,  3.7350972],
                                        [ 4.6573906, -3.7417417,  3.8269796],
                                        [ 5.2455406, -4.2887607,  4.0340977],
                                        [ 5.6874466, -4.78348  ,  4.3957458],
                                        [ 6.3070607, -5.4096227,  4.8495393],
                                        [ 6.957362 , -5.8742   ,  5.1154265],
                                        [ 7.3895497, -6.276678 ,  5.3706503],
                                        [ 7.9700212, -6.6828046,  5.871219 ],
                                        [ 8.681583 , -7.0160317,  6.3179407],
                                        [ 9.4586   , -7.356199 ,  6.684747 ]],
                                       dtype=np.float32))
