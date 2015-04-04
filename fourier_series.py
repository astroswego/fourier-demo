from argparse import ArgumentParser
from itertools import chain
from os import path
from numpy import arange, PINF, zeros
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import plotypus.lightcurve as lc
from plotypus.preprocessing import Fourier

def get_args():
    parser = ArgumentParser("fourier_series.py")

    parser.add_argument("-i", "--input", type=str,
        help="Table file containing time, magnitude, and (optional) error "
             "in its columns.")
    parser.add_argument("-o", "--output", type=str,
        default=".",
        help="Directory to output demo plots.")
    parser.add_argument("-n", "--name", type=str,
        default="",
        help="Name of star to use as prefix in output filenames")
    parser.add_argument("-t", "--type", type=str,
        default="png",
        help="File type to output plots in. Default is png.")
    parser.add_argument("-p", "--period", type=float,
        help="Period to phase observations by.")
    parser.add_argument("-d", "--fourier-degree", type=int, nargs=2,
        default=(1, 10), metavar=("MIN", "MAX"),
        help="Lower and upper bounds on degree of Fourier series. "
             "Defaults to [1, 10].")
    parser.add_argument("--use-cols", type=int, nargs="+",
        default=(0, 1, 2), metavar="C",
        help="Columns to read time, magnigude, and (optional) error from, "
             "respectively. "
             "Defaults to 0, 1, 2.")

    args = parser.parse_args()

    args.prefix = (args.name + "-") if args.name else ""

    return args

def main():
    args = get_args()

    predictor = lc.make_predictor(
        regressor=LinearRegression(fit_intercept=False),
        fourier_degree=args.fourier_degree,
        use_baart=True)

    result = lc.get_lightcurve_from_file(args.input, period=args.period,
                                         predictor=predictor,
                                         sigma=PINF)
    lightcurve   = result["lightcurve"]
    phased_data  = result["phased_data"]
    coefficients = result["coefficients"]
    model        = result["model"]
    degree       = result["degree"]
    shift        = result["shift"]

    phase_observed, mag_observed, *err = phased_data.T
    
    mag_min = min(lightcurve.min(), mag_observed.min())
    mag_max = max(lightcurve.max(), mag_observed.max())

    phases = arange(0, 1, 0.01)
    design_matrix = Fourier.design_matrix(phases+shift, degree)

    # Now multiply out individual columns from the design matrix
    # to make the separate harmonics
    raw_components = design_matrix * coefficients

    A_0       = raw_components[:, 0]
    harmonics = raw_components[:, 1::2] + raw_components[:, 2::2]
    components = chain([0], harmonics.T)

    partial_lc = zeros(100)

    for i, c in enumerate(components):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.scatter(phase_observed, mag_observed, color="k")
        if (partial_lc != 0).all():
            ax.plot(phases, partial_lc, "r-")
        else:
            partial_lc += A_0
        ax.plot(phases, c+A_0, "g--")

        ax.set_xlabel("Phase")
        ax.set_ylabel("Magnitude")
        ax.set_title("{} component".format(with_ordinal(i)))

        fig.savefig(
            path.join(args.output,
                      "{0}fourier-series-{1:02d}.{2}".format(args.prefix,
                                                             i,
                                                             args.type)))
        plt.close(fig)

        partial_lc += c

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(phase_observed, mag_observed, color="k")
    ax.plot(phases, partial_lc, "r-")

    ax.set_xlabel("Phase")
    ax.set_ylabel("Magnitude")
    ax.set_title("Complete Lightcurve")

    fig.savefig(path.join(args.output,
                          "{0}fourier-series-complete.{1}".format(args.prefix,
                                                                  args.type)))
    plt.close(fig)

    return 0


def with_ordinal(n):
    if 10 <= n % 100 < 20:
        return str(n) + "th"
    else:
       return  str(n) + {1 : "st", 2 : "nd", 3 : "rd"}.get(n % 10, "th")


if __name__ == "__main__":
    exit(main())
