import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.layout_engine import TightLayoutEngine
import numpy as np
import scipy
from cnvm.parameters import load_params


def fit_sin(tt, yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1] - tt[0]))  # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(
        ff[np.argmax(Fyy[1:]) + 1]
    )  # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.0**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.0 * np.pi * guess_freq, 0.0, guess_offset])

    def sinfunc(t, A, w, p, c):
        return A * np.sin(w * t + p) + c

    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    f = w / (2.0 * np.pi)
    fitfunc = lambda t: A * np.sin(w * t + p) + c
    return {
        "amp": A,
        "omega": w,
        "phase": p,
        "offset": c,
        "freq": f,
        "period": 1.0 / f,
        "fitfunc": fitfunc,
        "maxcov": np.max(pcov),
        "rawres": (guess, popt, pcov),
    }


def plot_cv_sinus():
    params = load_params("data/params.pkl")
    network = params.network
    alphas = np.load("data/cv_optim.npz")["alphas"]

    nodelist = [0]
    neighbors = [n for n in network.neighbors(0)]
    nodelist.append(neighbors[0])
    other_neighbor = neighbors[1]

    i = 1
    while nodelist[-1] != other_neighbor:
        neighbors = [n for n in network.neighbors(nodelist[i])]
        if neighbors[0] == nodelist[i - 1]:
            next_neighbor = neighbors[1]
        else:
            next_neighbor = neighbors[0]
        nodelist.append(next_neighbor)
        i = i + 1

    dim_cv = alphas.shape[1]
    fig, axes = plt.subplots(dim_cv, 2, figsize=(5, 5))

    pos = nx.kamada_kawai_layout(network)
    alphas /= np.max(np.abs(alphas))
    ylim = (-1.1, 1.1)

    # plot alphas
    labels = [
        r"$\Lambda_{1}$",
        r"$\Lambda_{2}$",
        r"$\Lambda_{3}$",
        r"$\Lambda_{4}$",
        r"$\Lambda_{5}$",
    ]
    for i in range(dim_cv):
        ax = axes[i, 0]
        this_alphas = alphas[:, i] / np.max(np.abs(alphas[:, i]))
        img = nx.draw_networkx_nodes(
            network,
            pos=pos,
            ax=ax,
            node_color=this_alphas,
            node_size=30,
            vmin=ylim[0],
            vmax=ylim[1],
        )
        nx.draw_networkx_edges(network, pos, ax=ax)
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_ticks([-1, 0, 1])

        cut = 1.15
        xmax = cut * max(xx for xx, yy in pos.values())
        ymax = cut * max(yy for xx, yy in pos.values())
        xmin = cut * min(xx for xx, yy in pos.values())
        ymin = cut * min(yy for xx, yy in pos.values())
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_ylabel(labels[i])
        # ax.text(0.05, 0.95, captions[i], transform=ax.transAxes, va='top', fontsize=caption_size)

    # plot sine
    for i in range(dim_cv):
        alpha = alphas[:, i] / np.max(np.abs(alphas[:, i]))
        alpha = alpha[nodelist]
        fit = fit_sin(np.arange(len(alpha)), alpha)
        omega = fit["omega"]
        axes[i, 1].plot(alpha, "-x", label=labels[i])
        if i != 0:
            axes[i, 1].plot(
                fit["fitfunc"](np.arange(len(alpha))),
                "-",
                label=rf"sin, $\omega={omega:.3f}$",
            )
        axes[i, 1].set_yticklabels([])
        if i < dim_cv - 1:
            axes[i, 1].set_xticklabels([])
        axes[i, 1].legend(loc="lower left")
        axes[i, 1].grid()
        axes[i, 1].set_ylim(ylim)
        # axes[i].set_ylabel(fr"$\xi_{i}$")
    axes[4, 1].set_xlabel("node")

    layout = TightLayoutEngine(pad=0)
    layout.execute(fig)

    plt.savefig("plots/ring_cv_sinus.pdf")
