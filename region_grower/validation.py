'''Module dedicated to validation. It is unused and incomplete so far'''
# Import space
# Check generated data
import tmd
from tmd import view

try:
    pass
except ImportError:
    raise ImportError(
        'region-grower[validate] is not installed. '
        'Please install it by doing: pip install region-grower[validate]')


def validate(output):
    '''Validate'''
    # 1. Check somata positions
    pop = tmd.io.load_population(output)
    # somata = np.array([n.soma.get_center() for n in pop.neurons])
    # somataR = np.array([n.soma.get_diameter() for n in pop.neurons])

    # mlab.points3d(somata[:,0], somata[:,1], somata[:,2], somataR,
    #               colormap='Reds', scale_factor=2.)

    # 2. Check subset of cells in space (2d)
    view.view.population(pop, title='')
    import matplotlib.pyplot as plt
    plt.show()
