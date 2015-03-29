from argparse import ArgumentParser
from os import path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d

import numpy
from plotypus.preprocessing import Fourier

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

from plotypus.utils import make_sure_path_exists

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def get_args():
    parser = ArgumentParser("fourier_space.py")

    parser.add_argument("-i", "--input", type=str,
        help="Table file containing time, magnitude, and (optional) error "
             "in its columns.")
    parser.add_argument("-o", "--output", type=str,
        default=".",
        help="Directory to output demo plots.")
    parser.add_argument("-t", "--type", type=str,
        default="png",
        help="File type to output plots in. Default is png.")
    parser.add_argument("-p", "--period", type=float,
        help="Period to phase observations by.")
    parser.add_argument("--use-cols", type=int, nargs="+",
        default=(0, 1, 2),
        help="Columns to read time, magnigude, and (optional) error from, "
             "respectively. "
             "Defaults to 0, 1, 2.")
    parser.add_argument("--2d", dest="two_dee",
        action="store_true",
        help="Enable 2D plot")
    parser.add_argument("--3d-flat", dest="three_dee_flat",
        action="store_true",
        help="Enable flat 3D plot")
    parser.add_argument("--3d-rotate", dest="three_dee_rotate",
        action="store_true",
        help="Enable rotating 3D plot")
    parser.add_argument("--3d-plane", dest="three_dee_plane",
        action="store_true",
        help="Enable rotating plane-fit plot")

    args = parser.parse_args()

    return args

def plot2d(x, y, x_label, y_label, filename,
           color='b', size=10, marker='.'):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(x, y,
               color=color, s=size, marker=marker)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_xticks((-1.0, -0.5, 0.0, 0.5, 1.0))
    ax.set_yticks((-1.0, -0.5, 0.0, 0.5, 1.0))

    fig.savefig(filename)

    plt.close(fig)

def plot3d(x, y, z, x_label, y_label, z_label, filename,
           color='b', size=10, marker='.'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_aspect('equal', 'datalim')

    ax.scatter(x, y, z,
               color=color, s=size, marker=marker)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    ax.set_xticks((-1.0, -0.5, 0.0, 0.5, 1.0))
    ax.set_yticks((-1.0, -0.5, 0.0, 0.5, 1.0))

    fig.savefig(filename)

    plt.close(fig)


def plot3drotate(x, y, z, x_label, y_label, z_label, file_prefix, file_type,
                 color='b', size=10, marker='.'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_aspect('equal', 'datalim')

    ax.scatter(x, y, z,
               color=color, s=size, marker=marker)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    ax.set_xticks((-1.0, -0.5, 0.0, 0.5, 1.0))
    ax.set_yticks((-1.0, -0.5, 0.0, 0.5, 1.0))

    for i in range(360):
        ax.view_init(azim=i)
        fig.savefig("{0}-{1:03d}.{2}".format(file_prefix, i, file_type))

    plt.close(fig)

def plot3dplane(x, y, z, A_0, a, b,
                x_label, y_label, z_label, file_prefix, file_type,
                color='b', size=10, marker='.'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_aspect('equal', 'datalim')

    xx, yy = numpy.meshgrid(numpy.arange(-1, 1, 0.01),
                            numpy.arange(-1, 1, 0.01))
    zz = A_0 + a*xx + b*yy

    mean_x, mean_y, mean_z = x.mean(), y.mean(), z.mean()
    mean_vec = numpy.array([mean_x, mean_y, mean_z])

    a_vec = numpy.array([1.0, 0.0, a])+mean_vec
    b_vec = numpy.array([0.0, 1.0, b])+mean_vec
    
    ax.scatter(x, y, z,
               color=color, s=size, marker=marker)
    ax.plot_surface(xx, yy, zz,
                    color="#555555", alpha=0.2)

    mean_x, mean_y, mean_z = x.mean(), y.mean(), z.mean()

    for vec in [a_vec, b_vec]:
        arrow = Arrow3D([mean_x, vec[0]],
                        [mean_y, vec[1]],
                        [mean_z, vec[2]],
                        mutation_scale=20,
                        lw=3, arrowstyle="-|>", color="r")
        ax.add_artist(arrow)
    

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    ax.set_xticks((-1.0, -0.5, 0.0, 0.5, 1.0))
    ax.set_yticks((-1.0, -0.5, 0.0, 0.5, 1.0))

    for i in range(360):
        ax.view_init(azim=i)
        fig.savefig("{0}-{1:03d}.{2}".format(file_prefix, i, file_type))

    plt.close(fig)


def main():
    args = get_args()

    make_sure_path_exists(args.output)

    phase, mag, *err = numpy.loadtxt(args.input, usecols=args.use_cols,
                                     unpack=True)

    design_matrix = Fourier.design_matrix(phase/args.period, 1)
    coeffs, *_ = numpy.linalg.lstsq(design_matrix, mag)
    A_0, a, b = coeffs

    if args.two_dee:
        plot2d(design_matrix[:,1], design_matrix[:,2],
               r"$\sin(1 \omega t)$", r"$\cos(1 \omega t)$",
               path.join(args.output, "2D-fourier-space." + args.type))

    if args.three_dee_flat:
        plot3d(design_matrix[:,1], design_matrix[:,2], mag,
               r"$\sin(1 \omega t)$", r"$\cos(1 \omega t)$", r"$m$",
               path.join(args.output, "3D-fourier-space." + args.type))

    if args.three_dee_rotate:
        plot3drotate(design_matrix[:,1], design_matrix[:,2], mag,
                     r"$\sin(1 \omega t)$", r"$\cos(1 \omega t)$", r"$m$",
                     path.join(args.output, "3D-fourier-space"), args.type)

    if args.three_dee_plane:
        plot3dplane(design_matrix[:,1], design_matrix[:,2], mag,
                    A_0, a, b,
                    r"$\sin(1 \omega t)$", r"$\cos(1 \omega t)$", r"$m$",
                    path.join(args.output, "3D-fourier-space-plane"), args.type)

    return 0

if __name__ == "__main__":
    exit(main())
