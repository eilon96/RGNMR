import numpy as np
import torch
from sympy import floor
from sympy.physics.quantum.sho1d import omega


def generate_matrix_torch(n1, n2, singular_values):
    """
    Generate a matrix with specific singular values using PyTorch.
    :param int n1: number of rows
    :param int n2: number of columns
    :param list singular_values: required singular values
    :return: Generated matrix with specified singular values.
    """
    rank = len(singular_values)
    U, _ = torch.linalg.qr(torch.randn(n1, rank))  # QR decomposition for orthonormal U
    V, _ = torch.linalg.qr(torch.randn(n2, rank))  # QR decomposition for orthonormal V
    D = torch.diag(torch.tensor(singular_values))
    return U @ D @ V.T

def generate_mask_torch(n1, n2, rank, p):
    """
    Generate a mask with at least 'rank' observed entries in each row and column using PyTorch.
    If 'p' is too small, the function might not return.

    :param int n1: number of rows
    :param int n2: number of columns
    :param int rank: rank of the matrix
    :param float p: probability of observing an entry
    :return: Binary observation mask (torch.Tensor of shape (n1, n2))
    """
    num_resamples = 0
    found = False
    p = max(0, min(p, 1))
    while not found:
        num_resamples += 1
        omega =  torch.round(0.5 * (torch.rand((n1, n2)) + p))  # Random mask based on probability p
        found = (omega.sum(dim=0).min() >= rank) and (omega.sum(dim=1).min() >= rank)  # Check row & col constraints

        if num_resamples % 10_000 == 0:
            print(f'Resampling mask {num_resamples}')

    return omega

def generate_mask(n1, n2, rank, oversampling_ratio):
    """
    Generate a mask with at least r observed entries in each row and column (r == rank)
    In case p is too small, function might not return
    :param int n1: number of rows
    :param int n2: number of columns
    :param float oversampling_ratio: the oversampling ratio the number of observed entries is (n1 + n2 - rank)*rank*oversampling_ratio
    :param int rank: rank of matrix
    """
    num_resamples = 0
    found = False
    while not found:
        num_resamples += 1

        # Choose random entries based on the oversampling ratio
        num_observed_entries = (n1 + n2 - rank)*rank*oversampling_ratio
        omega_idx = torch.randperm(n1*n2)[:num_observed_entries]
        i_Omega = np.mod(omega_idx, n1)
        j_Omega = omega_idx//n1

        # Create sparse matrix H with ones at outlier positions
        omega = np.zeros([n1, n2])
        omega[i_Omega, j_Omega] = 1
        # make sure there are enough visible entries on rows and columns
        found = (min(np.count_nonzero(omega, axis=0)) >= rank) and min(np.count_nonzero(omega, axis=1)) >= rank
        if (num_resamples % 1e4 == 0):
              print('resampling mask {}'.format(num_resamples))
    return omega


def generate_matrix(n1, n2,  singular_values):
    """
    Generate a matrix with specific singular values
    :param int n1: number of rows
    :param int n2: number of columns
    :param list singular_values: required singular values
    """
    rank = len(singular_values)
    U = np.random.randn(n1, rank)
    V = np.random.randn(n2, rank)
    U, _, _ = np.linalg.svd(U, full_matrices=False)
    V,_, _ = np.linalg.svd(V, full_matrices=False)
    D = np.diag(singular_values)
    return U @ D @ V.T


def generate_outliers(omega, alpha, rank):
    """
    Generate outliers for a sparse matrix omega based on a given fraction.

    :param torch.Tensor omega: Binary observation mask (entries are 0 or 1).
    :param float alpha: Fraction of observed entries to be marked as outliers.
    :param int rank: Minimum rank threshold for the matrix after introducing outliers.
    :return:
        - torch.Tensor: Outlier mask (same shape as omega), with 1s marking outlier locations.
        - int: Reject flag (0 if valid outlier mask is generated, 1 if rejected).
    """
    reject = 1
    reject_number = 0
    reject_max = 10
    m, n = omega.shape
    omega_torch = torch.from_numpy(omega)

    # Get the indices of non-zero entries in omega
    w = omega_torch.nonzero(as_tuple=True)

    while reject == 1:
        # Choose random outliers based on alpha
        omega_idx = torch.randperm(w[0].shape[0])[:int(alpha * w[0].shape[0])]
        i_Omega = w[0][omega_idx]
        j_Omega = w[1][omega_idx]

        # Create sparse matrix H with ones at outlier positions
        H = torch.zeros(m, n)
        H[i_Omega, j_Omega] = 1

        # Calculate nr_entr_col_omega_H and nr_entr_row_omega_H
        nr_entr_col_omega_H = torch.sum(omega_torch - H, dim=0)
        nr_entr_row_omega_H = torch.sum(omega_torch - H, dim=1)

        # Check if any column or row has fewer than `rank` entries after outliers
        if (torch.all(nr_entr_col_omega_H >= rank)) and (torch.all(nr_entr_row_omega_H >= rank)):
            reject = 0
        else:
            reject_number += 1
            if reject_number > reject_max:
                print('No mask found!')
                break

    return H, reject

def generate_test(n1, n2, rank, condition_number, oversampling_ratio, fraction_of_outliers):
    """
    Generate a synthetic low-rank plus sparse matrix decomposition test instance.

    :param int n1: Number of rows of the matrix.
    :param int n2: Number of columns of the matrix.
    :param int rank: Rank of the low-rank component.
    :param float condition_number: Condition number of the low-rank matrix (max_singular / min_singular).
    :param float oversampling_ratio: Ratio controlling the number of observed entries.
    :param float fraction_of_outliers: Fraction of observed entries that are corrupted.
    :return:
        - X (np.ndarray): Observed matrix (low-rank + sparse outliers on the mask).
        - L_star (np.ndarray): Ground truth low-rank matrix.
        - omega (torch.Tensor): Binary mask of observed entries.
        - int: Number of outlier entries.
    """

    singular_values = np.linspace(1, 1 / condition_number, rank)
    L_star = generate_matrix(n1, n2, singular_values)
    omega = generate_mask(n1, n2, rank, oversampling_ratio)
    outliers_mask, _ = generate_outliers(omega, fraction_of_outliers, rank)
    S_star = outliers_mask * (torch.rand_like(outliers_mask, dtype=float) * 2 * (np.max(abs(L_star))) - np.max(abs(L_star)))
    S_star = S_star.numpy()
    X = (L_star * omega + S_star)

    return X, L_star, omega, int(torch.sum(outliers_mask).item())