from argparse import ArgumentParser
from os import path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy
from plotypus.preprocessing import Fourier

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

from plotypus.utils import make_sure_path_exists

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
        help="Enable 2D plots")
    parser.add_argument("--3d-flat", dest="three_dee_flat",
        action="store_true",
        help="Enable flat 3D plots")
    parser.add_argument("--3d-rotate", dest="three_dee_rotate",
        action="store_true",
        help="Enable rotating 3D plots")

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


def main():
    args = get_args()

    make_sure_path_exists(args.output)

    phase, mag, *err = numpy.loadtxt(args.input, usecols=args.use_cols,
                                     unpack=True)

    design_matrix = Fourier.design_matrix(phase/args.period, 2)

    if args.two_dee:
        plot2d(design_matrix[:,1], design_matrix[:,2],
               r"$\sin(1 \omega t)$", r"$\cos(1 \omega t)$",
               path.join(args.output, "2D-11." + args.type))

        plot2d(design_matrix[:,1], design_matrix[:,3],
               r"$\sin(1 \omega t)$", r"$\sin(2 \omega t)$",
               path.join(args.output, "2D-12." + args.type))

    if args.three_dee_flat:
        plot3d(design_matrix[:,1], design_matrix[:,2], mag,
               r"$\sin(1 \omega t)$", r"$\cos(1 \omega t)$", r"$m$",
               path.join(args.output, "3D-11." + args.type))

        plot3d(design_matrix[:,1], design_matrix[:,3], mag,
               r"$\sin(1 \omega t)$", r"$\sin(2 \omega t)$", r"$m$",
               path.join(args.output, "3D-12." + args.type))

    if args.three_dee_rotate:
        plot3drotate(design_matrix[:,1], design_matrix[:,2], mag,
                     r"$\sin(1 \omega t)$", r"$\cos(1 \omega t)$", r"$m$",
                     path.join(args.output, "3D-11"), args.type)

        plot3drotate(design_matrix[:,1], design_matrix[:,3], mag,
                     r"$\sin(1 \omega t)$", r"$\sin(2 \omega t)$", r"$m$",
                     path.join(args.output, "3D-12"), args.type)

    return 0

if __name__ == "__main__":
    exit(main())
