

import string


def make_labels(axs, pos, pad=0):
    lbl = string.ascii_lowercase[pad:len(axs) + pad]

    for label, axis in zip(lbl, axs):
        axis.grid(True, which="both")
        axis.text(pos[0], pos[1], "({})".format(label), transform=axis.transAxes)

    return axs