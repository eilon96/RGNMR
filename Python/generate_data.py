import numpy as np
import torch


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

"""
Code taken from R2RILS by Jonathan Bauch, Boaz Nadler and Pini Zilber
"""

def generate_mask(n1, n2, rank, p):
    """
    Generate a mask with at least r observed entries in each row and column (r == rank)
    In case p is too small, function might not return
    :param int n1: number of rows
    :param int n2: number of columns
    :param float p: probability of observing an entry
    :param int rank: rank of matrix
    """
    num_resamples = 0
    found = False
    while not found:
        num_resamples += 1
        omega = np.round(0.5 * (np.random.random((n1, n2)) + p))
        # make sure there are enough visible entries on rows and columns
        found = (min(np.count_nonzero(omega, axis=0)) >= rank) and min(np.count_nonzero(omega, axis=1)) >= rank
        if (num_resamples % 1e4 == 0):
              print('resampling mask {}'.format(num_resamples))
    return omega


"""
Code taken from R2RILS by Jonathan Bauch, Boaz Nadler and Pini Zilber
"""

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
    Generate outliers for a sparse matrix omega based on alpha.

    Arguments:
    omega : torch.Tensor (m x n) sparse matrix
        The matrix with chosen entries (1s) and unchosen entries (0s).
    alpha : float
        Probability to choose an outlier.
    rank : int
        Minimum rank threshold for the matrix after outliers are introduced.

    Returns:
    H : torch.Tensor (m x n)
        The generated sparse matrix with outliers.
    reject : int
        Indicator (0 if mask is found, 1 if mask is rejected).
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

def generate_test(n1, n2 , rank, condition_number, oversampling_ratio, fraction_of_outliers):
    singular_values = np.linspace(1, 1 / condition_number, rank)
    L_star = generate_matrix(n1, n2, singular_values)
    p = oversampling_ratio * rank * (n1 + n2 - rank) / (n1 * n2)
    omega = generate_mask   (n1, n2, rank, p)
    outliers_mask, _ = generate_outliers(omega, fraction_of_outliers, rank)
    S_star = outliers_mask * (torch.rand_like(outliers_mask, dtype=float) * 2 * (np.max(abs(L_star))) - np.max(abs(L_star)))
    S_star = S_star.numpy()
    X = (L_star * omega + S_star)

    return X, L_star, omega, int(torch.sum(outliers_mask).item())