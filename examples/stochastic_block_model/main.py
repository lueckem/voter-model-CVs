from run_method import *
from plot import *
from validation import *
import time


def main():
    run_method()  # run the method, takes ~15 minutes on 16-core CPU
    plot()  # plot results
    validate()  # run and plot validation


def run_method():
    start = time.time()

    """ set up the network and parameters of the CNVM """
    setup_params()  # this creates the data file params.pkl

    """ sample the anchor points and run voter model simulations starting at the anchor points """
    sample_anchors_and_cnvm()  # this creates the data file x_data.npz

    """ approximate the transition manifold with a kernel-based algorithm """
    approximate_tm()  # this creates the data file xi.npz

    """ apply linear regression """
    linear_regression()  # this creates the data file cv_optim.npz

    end = time.time()
    print(f"Took {end - start} seconds.")


def plot():
    plot_network()  # plot of network, creates file plot_network
    plot3d()  # plot of transition manifold, creates file plot_tm
    plot_cv()  # plot of optimal cv, creates file plot_cv


def validate():
    # construct and plot three states x
    build_level_sets()
    plot_level_set()

    # validate their MMD and plot results
    validate_mmd()
    plot_mmd()


if __name__ == "__main__":
    main()
