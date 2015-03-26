from argparse import ArgumentParser
from os import path
import matplotlib.pyplot as plt

import numpy
from sklearn.linear_model import LinearRegression
from plotypus.lightcurve import get_lightcurve_from_file, make_predictor
from plotypus.utils import make_sure_path_exists

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


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
    parser.add_argument("-d", "--fourier-degree", type=int,
        help="Degree of fit")
    parser.add_argument("--use-cols", type=int, nargs="+",
        default=(0, 1, 2),
        help="Columns to read time, magnigude, and (optional) error from, "
             "respectively. "
             "Defaults to 0, 1, 2.")
    parser.add_argument("--radius", type=float, nargs=2,
        default=(0.2, 0.25), metavar=["R-MIN", "R-MAX"],
        help="Boundaries on radius of star visualization "
             "(default 0.2, 0.25)")
    parser.add_argument("--brightness", type=float, nargs=2,
        default=(0.7, 1.0), metavar=["B-MIN", "B-MAX"],
        help="Boundaries on brightness of star visualization on 0-1 scale "
             "(default 0.7, 1.0)")

    args = parser.parse_args()

    args.color = numpy.array(args.color)

    return args

def linear_map(x, x_min, x_max, y_min, y_max):
    return (x-x_min)*(y_max-y_min)/(x_max-x_min) + y_min

def display(index, phases, mags, output, file_type,
            mag_min=0.0, mag_max=1.0,
            radius_min=0.2, radius_max=0.25,
            alpha_min=0.1, alpha_max=0.7,
            color=numpy.array([1, 1, 0])):
    fig, axes = plt.subplots(1, 2)
    lc_axis, star_axis = axes

    lc_axis.invert_yaxis()

    star_axis.set_aspect("equal")
    star_axis.set_axis_bgcolor("black")
    star_axis.xaxis.set_visible(False)
    star_axis.yaxis.set_visible(False)

    phase, mag = phases[index], mags[index]
    rad = linear_map(mag, mag_max, mag_min, radius_min, radius_max)
    col = (1, 1, linear_map(mag, mag_max, mag_min, alpha_min, alpha_max))
    star = plt.Circle((0.5, 0.5), rad, color=col)

    lc_axis.plot(phases, mags, color="b")
    lc_axis.axvline(x=phase, linewidth=1, color="r")
    lc_axis.set_xlabel(r"$\phi$")
    lc_axis.set_ylabel(r"$m$")

    star_axis.add_artist(star)

    fig.savefig(path.join(output, "demo-{0:02d}.{1}".format(index, file_type)))
    plt.close(fig)

def main():
    args = get_args()

    make_sure_path_exists(args.output)

    predictor = make_predictor(
        regressor=LinearRegression(fit_intercept=False),
        fourier_degree=args.fourier_degree,
        use_baart=True)

    phases = numpy.arange(0, 1, 0.01)

    result = get_lightcurve_from_file(args.input, period=args.period,
                                      n_phases=100,
                                      predictor=predictor,
                                      sigma=numpy.PINF)

    lightcurve = result["lightcurve"]

    mag_min, mag_max = lightcurve.min(), lightcurve.max()

    for i in range(100):
        display(i, phases, lightcurve, args.output, args.type,
                mag_min=mag_min, mag_max=mag_max,
                radius_min=args.radius[0], radius_max=args.radius[1],
                color=args.color)

    return 0

if __name__ == "__main__":
    exit(main())
