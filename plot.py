
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from numpy import linalg as LA
import numpy as np

from plot_annotator import PlotAnnotator


def plot_pca_multi_2d(Xtrans, pca_components, filename_base, dimensions = 3, labels = None, pointlabels = None):
    for i, c in enumerate(np.linspace(0, 1, dimensions)):
        plot_pca_2d(
            Xtrans[:, i:(i+2)],
            pca_components[i:(i+2)],
            filename_base + str(i),
            labels = labels,
            pointlabels = pointlabels,
            color = cm.spectral(c)
            )

def plot_pca_2d(Xtrans, pca_components, filename_base, labels = None, pointlabels = None, color = 'blue'):
    """
    Xtrans: matrix transformed by the PCA
    pca_components: PCA vectors (pca.components_)
    """
    arrow_scale = 10

    # We'll only plot the first two compnonts
    Xtrans = Xtrans[:, :2]

    if labels is None:
        labels = ['x{}'.format(i) for i in range(pca_compnonets.shape[1])]

    plt.clf()

    plt.gcf().set_size_inches(20, 15)

    plt.scatter(Xtrans[:, 0], Xtrans[:, 1], marker = 'o', s = 0.5, color = color)
    ax = plt.gca()

    # Now draw the nice arrows
    # rows of pca.components_ are the individual components, columns
    # are features
    for feature, label in zip(pca_components.T * arrow_scale, labels):

        #print(LA.norm(feature[:2]))
        if LA.norm(feature[:2]) < (arrow_scale / 10.0 ):
            continue

        plt.arrow(0, 0, feature[0], feature[1],
            color = 'k', head_width = arrow_scale * 0.01,
            head_length = arrow_scale * 0.01, alpha = 0.5)

        sp0 = ax.transData.transform_point((0, 0))
        sp = ax.transData.transform_point(feature[:2])

        angle = np.arctan2(sp[1] - sp0[1], sp[0] - sp0[0]) * 180.0 / np.pi
        s = 1.5
        plt.text(feature[0] * s, feature[1] * s, label,
            color = 'k', ha = 'left', va = 'bottom',
            rotation = angle, rotation_mode = 'anchor')


    plt.xlabel('PC0')
    plt.ylabel('PC1')
    plt.axis('off')

    print('creating: {}'.format(filename_base + '.png'))
    plt.savefig(filename_base + '.png', bbox_inches = 'tight')

    #plt.show()


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot_pca_3d(Xtrans, pca_components, labels = None, pointlabels = None):
    """
    Xtrans: matrix transformed by the PCA
    pca_components: PCA vectors (pca.components_)
    """
    arrow_scale = 10

    # We'll only plot the first three compnonts
    Xtrans = Xtrans[:, :3]

    if labels is None:
        labels = ['x{}'.format(i) for i in range(pca_compnonets.shape[1])]

    fig = plt.figure()
    plt.clf()
    ax = Axes3D(fig, elev = 48, azim = 134)

    plt.cla()


    colors = np.random.rand(Xtrans.shape[0])
    ax.scatter(Xtrans[:, 0], Xtrans[:, 1], Xtrans[:, 2], c = colors, alpha = 0.5, gid = np.arange(Xtrans.shape[0]), picker = True)

    # Now draw the nice arrows
    # rows of pca.components_ are the individual components, columns
    # are features
    for feature, label in zip(pca_components.T * arrow_scale, labels):

        if LA.norm(feature[:2]) < (arrow_scale / 10.0 ):
            continue

        a = Arrow3D(
            [0, feature[0]], [0, feature[1]], [0, feature[2]],
            color = 'k',
            arrowstyle = '->',
            mutation_scale = 20,
            alpha = 0.5)
        ax.add_artist(a)
        ax.text(feature[0] * 1.15, feature[1] * 1.15, feature[2] * 1.15, label, color = 'k', ha = 'center', va = 'center')

    annotator = PlotAnnotator(ax, pointlabels)
    fig.canvas.mpl_connect('pick_event', annotator)
    plt.show()


