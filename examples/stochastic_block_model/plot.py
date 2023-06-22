import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.layout_engine import TightLayoutEngine
import matplotlib
import numpy as np
from cnvm.parameters import load_params


def plot3d():
    xi = np.load("data/xi.npy")

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.scatter(xi[:, 0], xi[:, 2], xi[:, 1], c=xi[:, 2])

    ax.set_xlabel(r"$\xi_1$")
    ax.set_ylabel(r"$\xi_3$")
    ax.set_zlabel(r"$\xi_2$")

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])

    ax.view_init(14, -73, 0)

    plt.tight_layout()
    plt.savefig("plots/plot_tm.pdf")


def plot_network():
    params = load_params("data/params.pkl")
    network = params.network

    communities = nx.community.greedy_modularity_communities(network)

    fig, ax = plt.subplots(figsize=(6, 5))
    pos = nx.spring_layout(network, seed=100, k=0.09)

    node_size = 70
    cmap = matplotlib.colormaps["viridis"]

    cluster0 = nx.draw_networkx_nodes(
        network,
        pos=pos,
        nodelist=list(communities[0]),
        ax=ax,
        node_size=node_size,
        node_color=cmap.colors[0],
    )
    cluster1 = nx.draw_networkx_nodes(
        network,
        pos=pos,
        nodelist=list(communities[1]),
        ax=ax,
        node_size=node_size,
        node_color=cmap.colors[-1],
    )
    cluster2 = nx.draw_networkx_nodes(
        network,
        pos=pos,
        nodelist=list(communities[2]),
        ax=ax,
        node_size=node_size,
        node_color=cmap.colors[127],
    )
    nx.draw_networkx_edges(network, pos=pos, ax=ax, width=3)

    cut = 1.05
    xmax = cut * max(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    xmin = cut * min(xx for xx, yy in pos.values())
    ymin = cut * min(yy for xx, yy in pos.values())
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.legend(
        handles=[cluster0, cluster2, cluster1],
        labels=["cluster 1", "cluster 2", "cluster 3"],
        prop={"size": 27},
    )
    plt.tight_layout()
    plt.savefig("plots/plot_network.pdf")


def plot_cv():
    plt.rcParams["font.size"] = 23
    caption_size = 28

    xi = np.load("data/xi.npy")
    params = load_params("data/params.pkl")
    network = params.network
    pos = nx.spring_layout(network, seed=100, k=0.09)
    alphas = np.load("data/cv_optim.npz")["alphas"]
    colors = np.load("data/cv_optim.npz")["xi_fit"]

    xi /= np.max(np.abs(xi))
    colors /= np.max(np.abs(colors))
    v_min, v_max = -1, 1

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))

    # figure 1
    ax = axes[0, 0]
    ax.plot(xi[:, 0], colors[:, 0], "o")
    ax.set_ylabel(r"$\bar{\varphi}_1$")
    ax.set_xlabel(r"$\varphi_1$")
    ax.grid()
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.text(0.05, 0.95, "(a)", transform=ax.transAxes, va="top", fontsize=caption_size)

    # figure 2-4
    captions = ["(b)", "(c)", "(d)"]
    labels = [r"$\Lambda_{1}$", r"$\Lambda_{2}$", r"$\Lambda_{3}$"]
    axs = [axes[0, 1], axes[1, 0], axes[1, 1]]
    for i in range(3):
        ax = axs[i]
        this_alphas = alphas[:, i] / np.max(np.abs(alphas[:, i]))
        img = nx.draw_networkx_nodes(
            network,
            pos=pos,
            ax=ax,
            node_color=this_alphas,
            node_size=50,
            vmin=v_min,
            vmax=v_max,
        )
        nx.draw_networkx_edges(network, pos, ax=ax)
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_ticks([-1, 0, 1])

        cut = 1.05
        xmax = cut * max(xx for xx, yy in pos.values())
        ymax = cut * max(yy for xx, yy in pos.values())
        xmin = cut * min(xx for xx, yy in pos.values())
        ymin = cut * min(yy for xx, yy in pos.values())
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        ax.set_xlabel(labels[i])
        ax.text(
            0.05,
            0.95,
            captions[i],
            transform=ax.transAxes,
            va="top",
            fontsize=caption_size,
        )

    layout = TightLayoutEngine(pad=0)
    layout.execute(fig)

    # make plot [0,0] smaller
    position = axes[0, 0].get_position()
    new_position = rescale_bbox(position, 0.8)
    axes[0, 0].set_position(new_position)

    fig.savefig(f"plots/plot_cv.pdf")


def rescale_bbox(
    bbox: Bbox,
    factor: float,
) -> Bbox:
    new_width = factor * bbox.width
    x0, x1 = bbox.intervalx[0], bbox.intervalx[0] + new_width
    y0, y1 = bbox.intervaly[0], bbox.intervaly[1]
    return Bbox(np.array([[x0, y0], [x1, y1]]))
