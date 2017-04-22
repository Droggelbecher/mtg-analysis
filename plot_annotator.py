
# based on: http://scipy-cookbook.readthedocs.io/items/Matplotlib_Interactive_Plotting.html

import matplotlib.pyplot as plt

class PlotAnnotator(object):

    def __init__(self, ax, labels):
        self.labels = labels
        self.text = ax.text2D(0.1, 0.1, "foo", fontsize = 16, transform = ax.transAxes)

    def __call__(self, event):
        print(event.name, event.ind, self.labels[event.ind[0]])
        self.text.set_text(self.labels[event.ind[0]])
        event.canvas.draw()
        #plt.pause(0.5)



