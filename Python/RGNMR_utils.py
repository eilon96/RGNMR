import torch
import numpy as np
from scipy.sparse import linalg, identity, coo_matrix, csr_matrix, identity
import gc

MAX_SIZE_FOR_VISUALIZATON_OF_ESTIMATED_MATRIX = 20
INIT_WITH_SVD = 0
INIT_WITH_RANDOM = 1
INIT_WITH_USER_DEFINED = 2

def threshold_operator(S, mask, alpha):
    """
    :param torch.Tensor S: input matrix (2D tensor)
    :param float a_col: fraction of largest absolute values to keep per column
    :param float a_row: fraction of largest absolute values to keep per row
    :return: thresholded matrix
    :rtype: torch.Tensor
    """

    n1, n2 = S.shape
    kcol = int(alpha * mask.sum() / n2)
    krow = int(alpha * mask.sum() / n1)

    # Get top-k indices per column
    _, colloc = torch.topk(S.abs(), kcol, dim=0)

    # Get top-k indices per row
    _, rowloc = torch.topk(S.abs().T, krow, dim=0)

    # Create binary masks
    mask_col = torch.zeros_like(S, dtype=torch.bool)
    mask_row = torch.zeros_like(S, dtype=torch.bool)

    mask_col.scatter_(0, colloc, 1)  # Set selected column indices to 1
    mask_row.scatter_(1, rowloc.T, 1)  # Set selected row indices to 1

    # remove the selected entries
    return S - S * mask_col * mask_row

def generate_sparse_matrix_entries(omega, rank):
    """
    :param ndarray omega: a one-zero array that masks the unobserved entries
    :param int rank: rank of the matrix
    :param int n1: number of rows
    :param int n2: number of columns
    :return: row_indices, col_indices
    """
    n1, n2 = omega.shape
    # Get indices of nonzero elements in omega
    j_idx, k_idx = torch.nonzero(torch.from_numpy(omega), as_tuple=True)

    # Compute total nonzero entries
    num_entries = len(j_idx)

    # Generate row indices (each nonzero entry contributes 2 * rank rows)
    row_indices = torch.arange(num_entries).repeat_interleave(2 * rank)

    # Generate interleaved column indices (U, V, U, V, ...)
    col_indices = torch.empty((2 * num_entries, rank))

    # Create indices for U entries
    U_indices = k_idx.unsqueeze(1) * rank + torch.arange(rank).repeat(num_entries, 1)

    # Create indices for V entries
    V_indices = (n2 + j_idx).unsqueeze(1) * rank + torch.arange(rank).repeat(num_entries, 1)

    # Interleave U and V indices
    col_indices[0::2] = U_indices
    col_indices[1::2] = V_indices

    return row_indices, col_indices.flatten()

def init_RGNMR(init_option, X, omega, rank, num_of_outliers, init_U= None, init_V= None):
    n1, n2 = X.shape
    # set the initial estimate
    if init_option == INIT_WITH_SVD:
      # applies a thresholding operator on X then applies svd
      U, _, V = linalg.svds(threshold_operator(torch.from_numpy(X), omega, num_of_outliers/omega.sum()).numpy(), k=rank, tol=1e-16)
      V = V.T
    elif init_option == INIT_WITH_RANDOM:
      U = np.random.randn(n1, rank)
      V = np.random.randn(n2, rank)
      U = np.linalg.qr(U)[0]
      V = np.linalg.qr(V)[0]
    else:
      U = init_U
      V = init_V

    # initial estimate
    L_hat = U @ V.T

    # construct an initial estimate of the  set of non-corrupted entries
    vectorize_X = vectorize_observed_matrix(X, omega)
    vectorize_X_hat = vectorize_observed_matrix(L_hat, omega)
    D = binary_weights(np.abs(vectorize_X_hat - vectorize_X), num_of_outliers)
    return U, V, L_hat, D


def generate_sparse_A(U, V, omega, row_entries, column_entries, num_visible_entries, n1, n2, rank):

    mask = torch.from_numpy(omega).nonzero(as_tuple=True)  # Get indices where omega is nonzero
    U_selected = U[mask[0]]  # Select relevant U rows
    V_selected = V[mask[1]]  # Select relevant V rows
    data_vector = np.concatenate((U_selected, V_selected), axis=1).flatten()  # Concatenate and flatten
    return csr_matrix(coo_matrix((data_vector, (row_entries, column_entries)),
                                 shape=(num_visible_entries, rank * (n1 + n2))))


def vectorize_observed_matrix(A, omega):

    """
    :param ndarray X: a matrix
    :param ndarray omega: a zero-one matrix of the same shape as X
    :return: a vector of size |omega| of the entries in X that are not zero in omega
    """

    return A[omega != 0]


def binary_weights(entriwise_residuals, number_of_outliers):

    """
    :param ndarray Ax: a vector of length n
    :param ndarray b: a vector of length n
    :param int number_of_outliers: number of outliers
    :return: a diagonal matrix D with ones and zeros on the diagonal.
             D_(i,i)=1 if the i'th value in abs(Ax - b) is
             one of the number_of_outliers largest absolute values in abs(Ax-b)
    """

    _, indices = torch.topk(torch.from_numpy(entriwise_residuals), number_of_outliers, largest=True, sorted=False)

    # Convert indices_p from tensor to list for later use
    indices_p = indices.tolist()

    # Create a diagonal matrix D with ones and zeros
    D = identity(len(entriwise_residuals), format='csr')
    D[indices_p, indices_p] = 0

    return D

def get_U_V_from_solution(x, rank, n1, n2):

    VT = x[:rank * n2].reshape(rank, n2, order='F')
    UT = x[rank * n2:].reshape(rank, n1, order='F')
    return UT.T, VT.T

def solve_LSQR_problem(X, U, V, omega, D, sparse_matrix_rows, sparse_matrix_columns, tol, max_iterations):
  """
  At iteration t RGNMR solves
              (U_next, V_next) = min_{U^, V^} ||U@V^.T + U^@V.T - U@V.T - X||_{F(\Omega \cap \Lambda_{t})}
  We construct this problem as weighted least of squares problem min ||DAx-Db||.
  The matrix D define binary weights, with zeros on the entries that are estimated to be corrupted,
  and ones otherwise.

  :param ndarray U: our current U estimate,  shape (n1,rank)
  :param ndarray V: our current V estimate,  shape (n2,rank)
  :param ndarray omega: a zero-one matrix of the same shape as X, with ones in the observed entries
  :param ndarray D: a diagonal matrix that has ones and zeros on the diagonal.
  :param ndarray sparse_matrix_rows: used to contruct the lsqr problem
  :param ndarray sparse_matrix_columns: used to contruct the lsqr problem
  :param float tol: tolerance for the lsqr solver
  :param int max_iterations: maximum number of iterations for the lsqr solver

  :return the next estimates U_next, V_next, L_hat
  and a vector entriwise_residual of the residuals of the new estimate L_hat from the observed matrix
  """

  n1, rank = U.shape
  n2, _ = V.shape
  num_visible_entries = np.count_nonzero(omega)

  ## build the least of squares problem
  # A is a sparse matrix of shape (|Omega|,(n1+n2)*rank)
  A = generate_sparse_A(U, V, omega, sparse_matrix_rows, sparse_matrix_columns, num_visible_entries, n1, n2, rank)

  # b is a vecorization of the non zero entries in omega*(X  + U@V.T) with shape (|omega|,)
  b = vectorize_observed_matrix(X + U @ V.T, omega)

  # solve the trimmed least squares problem, x is the solution , res = ||DAx - Db||
  x, _, _, res = linalg.lsqr(D@A, D@b, atol=tol, btol=tol, iter_lim=max_iterations)[:4]

  # obtain new estimates for U and V from x
  U_next, V_next = get_U_V_from_solution(x, rank, n1, n2)

  # obtain new estimate for X
  L_hat = U @ V_next.T + U_next @ V.T - U @ V.T
  # obtain a vectorize estimate of the underlying matrix
  entriwise_residuals = np.abs(A@x - b)

  relRes = res/np.linalg.norm(D@b)

  # A can get very large, we therefore free memory by deleting it
  del A
  gc.collect()

  return U_next, V_next, L_hat, entriwise_residuals, relRes

def report_RGNMR_progression( verbose, iter_num, relRes):

  if verbose:
    print("[INSIDE RGNMR] iter: " + str(iter_num) + ", relRes: " + str(relRes))

def check_early_stopping_criteria(early_stopping_flag, relRes, stop_relRes, all_relRes,
                                  stop_relDiff, X_hat, X_hat_previous, stop_Lambda_converged, iterations_since_Lambda_changed,
                                  stop_relResDiff, verbose):

  if stop_relRes > 0:
    early_stopping_flag |= relRes < stop_relRes
  if stop_relDiff > 0:
    X_hat_diff =  np.linalg.norm(X_hat - X_hat_previous, ord='fro') / np.linalg.norm(X_hat, ord='fro')
    early_stopping_flag |= X_hat_diff < stop_relDiff
  if stop_Lambda_converged:
      early_stopping_flag |= iterations_since_Lambda_changed > 1
  if stop_relResDiff > 0:
    early_stopping_flag |= np.abs(relRes / all_relRes[-2] - 1) < stop_relResDiff

  if verbose and early_stopping_flag:
    print("[INSIDE RGNMR] early stopping")
  return early_stopping_flag
