
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from numpy import linalg as LA
import numpy as np

from plot_annotator import PlotAnnotator

def plot_pca_2d(Xtrans, pca_components, labels = None, pointlabels = None):
    """
    Xtrans: matrix transformed by the PCA
    pca_components: PCA vectors (pca.components_)
    """
    arrow_scale = 10

    # We'll only plot the first two compnonts
    Xtrans = Xtrans[:, :2]

    if labels is None:
        labels = ['x{}'.format(i) for i in range(pca_compnonets.shape[1])]


    plt.scatter(Xtrans[:, 0], Xtrans[:, 1], marker = 'o', s = 0.5)

    # Now draw the nice arrows
    # rows of pca.components_ are the individual components, columns
    # are features
    for feature, label in zip(pca_components.T * arrow_scale, labels):

        #print(LA.norm(feature[:2]))
        if LA.norm(feature[:2]) < (arrow_scale / 10.0 ):
            continue

        plt.arrow(0, 0, feature[0], feature[1], color = 'k', head_width = arrow_scale * 0.01, head_length = arrow_scale * 0.01, alpha = 0.5)
        plt.text(feature[0] * 1.15, feature[1] * 1.15, label, color = 'k', ha = 'center', va = 'center')

    plt.xlabel('PC0')
    plt.ylabel('PC1')

    plt.show()


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

    #if pointlabels:
        #for pos, label in zip(Xtrans, pointlabels):
            #ax.text(pos[0], pos[1], pos[2], label, color = 'k',alpha = 0.5)


    # Now draw the nice arrows
    # rows of pca.components_ are the individual components, columns
    # are features
    for feature, label in zip(pca_components.T * arrow_scale, labels):

        #print(LA.norm(feature[:2]))
        if LA.norm(feature[:2]) < (arrow_scale / 10.0 ):
            continue

        a = Arrow3D(
            [0, feature[0]], [0, feature[1]], [0, feature[2]],
            color = 'k',
            arrowstyle = '->',
            mutation_scale = 20,
            #head_width = arrow_scale * 0.01, head_length = arrow_scale * 0.01,
            alpha = 0.5)
        ax.add_artist(a)
        ax.text(feature[0] * 1.15, feature[1] * 1.15, feature[2] * 1.15, label, color = 'k', ha = 'center', va = 'center')

    annotator = PlotAnnotator(ax, pointlabels)

    #fig.canvas.mpl_connect('motion_notify_event', annotator)
    fig.canvas.mpl_connect('pick_event', annotator)

    plt.show()


