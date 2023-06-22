from interpretable_cvs.validation import (
    sample_state_like_x,
    plot_x_levelset,
    mmd_from_trajs,
    plot_validation_mmd,
)
from cnvm.parameters import load_params
from cnvm.utils import sample_many_runs
import pickle
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


def build_level_sets():
    params = load_params("data/params.pkl")
    with open("data/cv.pkl", "rb") as file:
        cv = pickle.load(file)

    for c in cv.collective_variables:
        c.normalize = False

    communities = nx.community.greedy_modularity_communities(params.network)

    share_cluster1 = 0.1
    share_cluster2 = 0.5

    nodes_cluster1 = list(communities[0])
    nodes_cluster2 = list(communities[1])

    x = np.zeros((2, params.num_agents), dtype=int)
    idx_cluster1 = np.random.choice(
        nodes_cluster1, int(share_cluster1 * params.num_agents / 3), replace=False
    )
    idx_cluster2 = np.random.choice(
        nodes_cluster2, int(share_cluster2 * params.num_agents / 3), replace=False
    )
    x[0, idx_cluster1] = 1
    x[0, idx_cluster2] = 1
    x[1] = np.copy(x[0])
    np.random.shuffle(x[1])

    weights = np.array(
        [
            cv.collective_variables[i].weights
            for i in range(len(cv.collective_variables))
        ]
    )

    x2 = sample_state_like_x(x[0], weights, num_samples=1)
    x = np.concatenate((x2, x))
    np.save("data/x_levelset.npy", x)


def validate_mmd():
    print("Validating...")
    t_max = 200
    num_samples = 1000
    num_timesteps = 100

    params = load_params("data/params.pkl")
    with open("data/cv.pkl", "rb") as file:
        cv = pickle.load(file)

    x_init = np.load("data/x_levelset.npy")[:3]
    print("CV values of the three states:")
    print(cv(x_init))

    t, x = sample_many_runs(
        params,
        x_init,
        t_max,
        num_timesteps,
        num_runs=num_samples,
        n_jobs=-1,
    )

    c = np.zeros((3, num_samples, num_timesteps, cv.dimension))
    for i in [0, 1, 2]:
        for j in range(num_samples):
            c[i, j] = cv(x[i, j])

    np.savez_compressed("data/data_validate_full.npz", t=t, c=x)
    np.savez_compressed("data/data_validate.npz", t=t, c=c)

    mmd = mmd_from_trajs(x)
    np.save("data/mmd_validate_full.npy", mmd)
    mmd = mmd_from_trajs(c)
    np.save("data/mmd_validate.npy", mmd)


def plot_mmd():
    mmd = np.load("data/mmd_validate.npy")
    mmd_full = np.load("data/mmd_validate_full.npy")
    t = np.load("data/data_validate.npz")["t"]

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    plot_validation_mmd(t, mmd, axes[0], r"MMD$_\varphi$")
    plot_validation_mmd(t, mmd_full, axes[1], r"MMD")

    fig.tight_layout()
    fig.savefig("plots/validate_mmd.pdf")


def plot_level_set():
    x = np.load("data/x_levelset.npy")
    network = load_params("data/params.pkl").network
    fig = plot_x_levelset(x, network)
    fig.tight_layout()
    fig.savefig("plots/level_set.pdf")
