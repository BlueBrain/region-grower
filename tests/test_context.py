import mock
import numpy as np
from pathlib import Path

from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_raises, assert_equal, assert_almost_equal

from voxcell.nexus.voxelbrain import Atlas
from region_grower.context import SpaceContext

DATA = Path(__file__).parent / 'data'

from mock import MagicMock

def test_get_layer():
    context = SpaceContext(MagicMock(),
                           DATA / 'distributions.json',
                           DATA / 'parameters.json')

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
                           DATA / 'distributions.json',
                           DATA / 'parameters.json')
    context._get_orientation = MagicMock(return_value = [-0.33126975, -0.91100982, -0.2456043 ])
    context._cumulative_thicknesses = MagicMock(return_value = [ 125, 225, 450, 575, 900, 1250])
    context._get_layer = MagicMock(return_value = 1)

    result = context.synthesize([6001.39477031, 772.6250185, 3778.3698741], 'L2_TPC:A')

    assert_array_almost_equal(result.apical_points, [[-14.207288, -34.351606,  -9.616721]])

    # This tests that input orientations are not mutated by the synthesize() call
    assert_array_almost_equal(context.tmd_parameters['L2_TPC:A']['apical']['orientation'],
                              [[0.0, 1.0, 0.0]])

    assert_array_almost_equal(result.neuron.soma.points,
                              np.array([[ 9.104976 ,  -0.8350304,  0.,      0.],
                                        [ 0.1265565,  9.05724   ,  0.,      0.],
                                        [-4.1508913,  7.8628845 ,  0.,      0.],
                                        [-9.0374   ,  1.3868117 ,  0.,      0.],
                                        [-9.12136  ,  0.34562492,  0.,      0.],
                                        [-3.028861 , -8.329533  ,  0.,      0.],
                                        [ 4.670966 , -7.8600206 ,  0.,      0.]],
                                       dtype=np.float32))

    assert_equal(len(result.neuron.root_sections), 7)
    assert_array_almost_equal(result.neuron.neurites[0].root_node.points,
                              np.array(
                                  [[ 0.12655652,  9.05724 ,    1.2442882,  0.3],
                                   [ 0.1398302 , 10.007194,    1.3747933,  0.3],
                                   [ 0.2490275 , 10.898591,   1.37083380,  0.3]],
                                       dtype=np.float32))


def test_context_external_diametrizer():
    np.random.seed(0)

    atlas = MagicMock()
    context = SpaceContext(atlas,
                           DATA / 'distributions_external_diametrizer.json',
                           DATA / 'parameters_external_diametrizer.json')
    context._get_orientation = MagicMock(return_value = [-0.33126975, -0.91100982, -0.2456043 ])
    context._cumulative_thicknesses = MagicMock(return_value = [ 125, 225, 450, 575, 900, 1250])
    context._get_layer = MagicMock(return_value = 1)

    result = context.synthesize([6001.39477031, 772.6250185, 3778.3698741], 'L2_TPC:A')
    assert_array_almost_equal(result.apical_points, [[-14.207288, -34.351606,  -9.616721]])

    # This tests that input orientations are not mutated by the synthesize() call
    assert_array_almost_equal(context.tmd_parameters['L2_TPC:A']['apical']['orientation'],
                              [[0.0, 1.0, 0.0]])

    assert_array_almost_equal(result.neuron.soma.points,
                              np.array([[ 9.104976 ,  -0.8350304,  0.,       0.],
                                        [ 0.1265565,  9.05724   ,  0.,       0.],
                                        [-4.1508913,  7.8628845 ,  0.,       0.],
                                        [-9.0374   ,  1.3868117 ,  0.,       0.],
                                        [-9.12136  ,  0.34562492,  0.,       0.],
                                        [-3.028861 , -8.329533  ,  0.,       0.],
                                        [ 4.670966 , -7.8600206 ,  0.,       0.]],
                                       dtype=np.float32))


    assert_equal(len(result.neuron.root_sections), 7)
    assert_array_almost_equal(result.neuron.neurites[0].root_node.points,
                              np.array(
                                  [[ 0.12655652,  9.05724 ,    1.2442882,  0.5201028],
                                   [ 0.1398302 , 10.007194,    1.3747933,  0.5162242],
                                   [ 0.2490275 , 10.898591,   1.3708338,  0.5125918]],
                                       dtype=np.float32))
