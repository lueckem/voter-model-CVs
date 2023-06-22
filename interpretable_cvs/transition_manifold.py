import numpy as np
from numba import njit, prange
import scipy.sparse.linalg as sla


class TransitionManifold:
    def __init__(
        self,
        bandwidth_transitions: float,
        bandwidth_diffusion_map: float,
        dimension: int = 10,
    ):
        self.bandwidth_transitions = bandwidth_transitions
        self.bandwidth_diffusion_map = bandwidth_diffusion_map
        self.dimension = dimension
        self.eigenvalues = None

    def fit(self, x_samples: np.ndarray):
        """
        Parameters
        ----------
        x_samples : np.ndarray
            Array containing the endpoints of num_samples simulations for each anchor point.
            Shape = (num_anchor_points, num_samples, dimension).

        Returns
        -------
        np.ndarray
            Array containing the coordinates of each anchor point in diffusion space.
            Shape = (num_anchor_points, dimension).
        """
        distance_matrix, _ = _numba_dist_matrix_gaussian_kernel(
            x_samples, self.bandwidth_transitions
        )
        return self._calc_diffusion_map(distance_matrix)

    def _calc_diffusion_map(self, distance_matrix: np.ndarray):
        eigenvalues, eigenvectors = calc_diffusion_maps(
            distance_matrix, self.dimension, self.bandwidth_diffusion_map
        )
        self.eigenvalues = eigenvalues
        return eigenvectors.real[:, 1:] * eigenvalues.real[np.newaxis, 1:]


@njit(parallel=True)
def _numba_dist_matrix_gaussian_kernel(
    x_samples: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    x_samples : np.ndarray
        Shape = (num_anchor_points, num_samples, dimension).
    sigma : float
        Bandwidth.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        1) Distance matrix, shape = (num_anchor_points, num_anchor_points).
        2) kernel matrix diagonal, shape = (num_anchor_points,)
    """
    num_anchor, num_samples, dimension = x_samples.shape

    # compute symmetric kernel evaluations
    kernel_diagonal = np.zeros(num_anchor)
    for i in range(num_anchor):
        kernel_diagonal[i] = _numba_gaussian_kernel_eval(
            x_samples[i], x_samples[i], sigma
        )

    # compute asymmetric kernel evaluations and assemble distance matrix
    distance_matrix = np.zeros((num_anchor, num_anchor))
    for i in prange(num_anchor):
        for j in range(i):
            this_sum = _numba_gaussian_kernel_eval(x_samples[i], x_samples[j], sigma)
            distance_matrix[i, j] = (
                kernel_diagonal[i] + kernel_diagonal[j] - 2 * this_sum
            )
    distance_matrix /= num_samples**2
    return distance_matrix + np.transpose(distance_matrix), kernel_diagonal


@njit(parallel=False)
def _numba_gaussian_kernel_eval(x: np.ndarray, y: np.ndarray, sigma: float):
    """
    Parameters
    ----------
    x : np.ndarray
        shape = (# x points, dimension)
    y : np.ndarray
        shape = (# y points, dimension)
    sigma : float
        bandwidth

    Returns
    -------
    float
        sum of kernel matrix
    """
    nx = x.shape[0]
    ny = y.shape[0]

    X = np.sum(x * x, axis=1).reshape((nx, 1)) * np.ones((1, ny))
    Y = np.sum(y * y, axis=1) * np.ones((nx, 1))
    out = X + Y - 2 * np.dot(x, y.T)
    out /= -(sigma**2)
    np.exp(out, out)
    return np.sum(out)


def calc_diffusion_maps(
    distance_matrix: np.ndarray, num_components: int, sigma: float, alpha: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve diffusion map eigenproblem

    Parameters
    ----------
    distance_matrix : np.ndarray
        Symmetric array of shape (n_features, n_features).
    num_components : int
        Number of eigenvalues and eigenvectors to compute.
    sigma : float
        Diffusion map kernel bandwidth parameter.
    alpha : float
        Diffusion map normalization parameter.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (eigenvalues, eigenvectors).
    """

    num_points = distance_matrix.shape[0]
    kernel_matrix = np.exp(-(distance_matrix**2) / sigma**2)
    row_sum = np.sum(kernel_matrix, axis=0)

    # compensating for testpoint density
    kernel_matrix = kernel_matrix / np.outer(row_sum**alpha, row_sum**alpha)

    # row normalization
    kernel_matrix = kernel_matrix / np.tile(sum(kernel_matrix, 0), (num_points, 1))

    # weight matrix
    weight_matrix = np.diag(sum(kernel_matrix, 0))

    # solve the diffusion maps eigenproblem
    return sla.eigs(kernel_matrix, num_components + 1, weight_matrix)


def main():
    x_samples = np.random.random((100, 10, 2))
    tm = TransitionManifold(1, 1, 3)
    tm.fit(x_samples)


if __name__ == "__main__":
    main()
