import numpy as np
from scipy import linalg
from typing import Union
import pykeops.torch as keops
import torch
import tqdm

emb_scale = 6

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat


def calculate_matching_score(embedding1, embedding2, sum_all=False):
    assert len(embedding1.shape) == 2
    assert embedding1.shape[0] == embedding2.shape[0]
    assert embedding1.shape[1] == embedding2.shape[1]

    dist = linalg.norm(embedding1 - embedding2, axis=1)
    if sum_all:
        return dist.sum(axis=0)
    else:
        return dist

def calculate_activation_statistics(activations):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    activations = activations * emb_scale
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    activation = activation * emb_scale
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm((activation[first_indices] - activation[second_indices])/2, axis=1)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()

def calculate_wasserstein(x: torch.Tensor, y: torch.Tensor, p: float = 2,
                            w_x: Union[torch.Tensor, None] = None,
                            w_y: Union[torch.Tensor, None] = None,
                            eps: float = 1e-3,
                            max_iters: int = 100, stop_thresh: float = 1e-5,
                            verbose=False):
    """
    Compute the Entropy-Regularized p-Wasserstein Distance between two d-dimensional point clouds
    using the Sinkhorn scaling algorithm. This code will use the GPU if you pass in GPU tensors.
    Note that this algorithm can be backpropped through
    (though this may be slow if using many iterations).

    :param x: A [n, d] tensor representing a d-dimensional point cloud with n points (one per row)
    :param y: A [m, d] tensor representing a d-dimensional point cloud with m points (one per row)
    :param p: Which norm to use. Must be an integer greater than 0.
    :param w_x: A [n,] shaped tensor of optional weights for the points x (None for uniform weights). Note that these must sum to the same value as w_y. Default is None.
    :param w_y: A [m,] shaped tensor of optional weights for the points y (None for uniform weights). Note that these must sum to the same value as w_y. Default is None.
    :param eps: The reciprocal of the sinkhorn entropy regularization parameter.
    :param max_iters: The maximum number of Sinkhorn iterations to perform.
    :param stop_thresh: Stop if the maximum change in the parameters is below this amount
    :param verbose: Print iterations
    :return: a triple (d, corrs_x_to_y, corr_y_to_x) where:
      * d is the approximate p-wasserstein distance between point clouds x and y
      * corrs_x_to_y is a [n,]-shaped tensor where corrs_x_to_y[i] is the index of the approximate correspondence in point cloud y of point x[i] (i.e. x[i] and y[corrs_x_to_y[i]] are a corresponding pair)
      * corrs_y_to_x is a [m,]-shaped tensor where corrs_y_to_x[i] is the index of the approximate correspondence in point cloud x of point y[j] (i.e. y[j] and x[corrs_y_to_x[j]] are a corresponding pair)
    """

    if not isinstance(p, int):
        raise TypeError(f"p must be an integer greater than 0, got {p}")
    if p <= 0:
        raise ValueError(f"p must be an integer greater than 0, got {p}")

    if eps <= 0:
        raise ValueError("Entropy regularization term eps must be > 0")

    if not isinstance(p, int):
        raise TypeError(f"max_iters must be an integer > 0, got {max_iters}")
    if max_iters <= 0:
        raise ValueError(f"max_iters must be an integer > 0, got {max_iters}")

    if not isinstance(stop_thresh, float):
        raise TypeError(f"stop_thresh must be a float, got {stop_thresh}")

    if len(x.shape) != 2:
        raise ValueError(f"x must be an [n, d] tensor but got shape {x.shape}")
    if len(y.shape) != 2:
        raise ValueError(f"x must be an [m, d] tensor but got shape {y.shape}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"x and y must match in the last dimension (i.e. x.shape=[n, d], "
                         f"y.shape[m, d]) but got x.shape = {x.shape}, y.shape={y.shape}")

    if w_x is not None:
        if w_y is None:
            raise ValueError("If w_x is not None, w_y must also be not None")
        if len(w_x.shape) > 1:
            w_x = w_x.squeeze()
        if len(w_x.shape) != 1:
            raise ValueError(f"w_x must have shape [n,] or [n, 1] "
                             f"where x.shape = [n, d], but got w_x.shape = {w_x.shape}")
        if w_x.shape[0] != x.shape[0]:
            raise ValueError(f"w_x must match the shape of x in dimension 0 but got "
                             f"x.shape = {x.shape} and w_x.shape = {w_x.shape}")
    if w_y is not None:
        if w_x is None:
            raise ValueError("If w_y is not None, w_x must also be not None")
        if len(w_y.shape) > 1:
            w_y = w_y.squeeze()
        if len(w_y.shape) != 1:
            raise ValueError(f"w_y must have shape [n,] or [n, 1] "
                             f"where x.shape = [n, d], but got w_y.shape = {w_y.shape}")
        if w_x.shape[0] != x.shape[0]:
            raise ValueError(f"w_y must match the shape of y in dimension 0 but got "
                             f"y.shape = {y.shape} and w_y.shape = {w_y.shape}")


    # Distance matrix [n, m]
    x_i = keops.Vi(x)  # [n, 1, d]
    y_j = keops.Vj(y)  # [i, m, d]
    if p == 1:
        M_ij = ((x_i - y_j) ** p).abs().sum(dim=2) # [n, m]
    else:
        M_ij = ((x_i - y_j) ** p).sum(dim=2) ** (1.0 / p)  # [n, m]

    # Weights [n,] and [m,]
    if w_x is None and w_y is None:
        w_x = torch.ones(x.shape[0]).to(x) / x.shape[0]
        w_y = torch.ones(y.shape[0]).to(x) / y.shape[0]
        w_y *= (w_x.shape[0] / w_y.shape[0])

    sum_w_x = w_x.sum().item()
    sum_w_y = w_y.sum().item()
    if abs(sum_w_x - sum_w_y) > 1e-5:
        raise ValueError(f"Weights w_x and w_y do not sum to the same value, "
                         f"got w_x.sum() = {sum_w_x} and w_y.sum() = {sum_w_y} "
                         f"(absolute difference = {abs(sum_w_x - sum_w_y)}")

    log_a = torch.log(w_x)  # [n]
    log_b = torch.log(w_y)  # [m]

    # Initialize the iteration with the change of variable
    u = torch.zeros_like(w_x)
    v = eps * torch.log(w_y)

    u_i = keops.Vi(u.unsqueeze(-1))
    v_j = keops.Vj(v.unsqueeze(-1))

    if verbose:
        pbar = tqdm.trange(max_iters)
    else:
        pbar = range(max_iters)

    for _ in pbar:
        u_prev = u
        v_prev = v

        summand_u = (-M_ij + v_j) / eps
        u = eps * (log_a - summand_u.logsumexp(dim=1).squeeze())
        u_i = keops.Vi(u.unsqueeze(-1))

        summand_v = (-M_ij + u_i) / eps
        v = eps * (log_b - summand_v.logsumexp(dim=0).squeeze())
        v_j = keops.Vj(v.unsqueeze(-1))

        max_err_u = torch.max(torch.abs(u_prev-u))
        max_err_v = torch.max(torch.abs(v_prev-v))
        if verbose:
            pbar.set_postfix({"Current Max Error": max(max_err_u, max_err_v).item()})
        if max_err_u < stop_thresh and max_err_v < stop_thresh:
            break

    P_ij = ((-M_ij + u_i + v_j) / eps).exp()

    approx_corr_1 = P_ij.argmax(dim=1).squeeze(-1)
    approx_corr_2 = P_ij.argmax(dim=0).squeeze(-1)

    if u.shape[0] > v.shape[0]:
        distance = (P_ij * M_ij).sum(dim=1).sum()
    else:
        distance = (P_ij * M_ij).sum(dim=0).sum()
    return distance, approx_corr_1, approx_corr_2
