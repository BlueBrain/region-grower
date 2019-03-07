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
                              np.array([[-6.2937384 ,  0.23849252,  0.        ],
                                        [-2.089914  , -5.7473774 ,  0.        ],
                                        [ 4.181802  , -2.9605203 ,  0.        ],
                                        [ 0.08732622,  6.2494955 ,  0.        ],
                                        [-0.84500784,  6.251952  ,  0.        ],
                                        [-2.8641052 ,  5.425396  ,  0.        ],
                                        [-4.5985866 ,  4.3190207 ,  0.        ],
                                        [-6.010247  ,  1.9177774 ,  0.        ]],
                                       dtype=np.float32))

    assert_equal(len(neuron.root_sections), 7)
    assert_array_almost_equal(neuron.root_sections[0].points,
                              np.array([[ 4.181802 , -2.9605203,  3.680869 ],
                                        [ 4.4350734, -2.996226 ,  3.793232 ],
                                        [ 5.0802574, -3.2453353,  4.0145335],
                                        [ 5.717802 , -3.3482637,  4.3777814],
                                        [ 6.5341225, -3.6494315,  4.8251734],
                                        [ 7.2887597, -3.8344555,  5.3626204],
                                        [ 7.9828873, -4.1905737,  5.702718 ],
                                        [ 8.53841  , -4.3957295,  5.9720254]], dtype=np.float32))


    assert_array_almost_equal(neuron.root_sections[0].diameters,
                              np.array([1.058735, 1.052401, 1.035936, 1.019135, 0.99695 , 0.975528,
                                        0.956231, 0.94148 ], dtype=np.float32))
