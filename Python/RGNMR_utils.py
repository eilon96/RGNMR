import torch 
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, identity

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

def generate_sparse_matrix_entries(omega, rank, n1, n2):
    """
    :param ndarray omega: a one-zero array that masks the unobserved entries
    :param int rank: rank of the matrix
    :param int n1: number of rows
    :param int n2: number of columns
    :return: row_indices, col_indices
    """
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


def generate_sparse_A(U, V, omega, row_entries, column_entries, num_visible_entries, n1, n2, rank):

    mask = torch.from_numpy(omega).nonzero(as_tuple=True)  # Get indices where omega is nonzero
    U_selected = U[mask[0]]  # Select relevant U rows
    V_selected = V[mask[1]]  # Select relevant V rows
    data_vector = np.concatenate((U_selected, V_selected), axis=1).flatten()  # Concatenate and flatten
    return csr_matrix(coo_matrix((data_vector, (row_entries, column_entries)),
                                 shape=(num_visible_entries, rank * (n1 + n2))))


def generate_b(X, omega):
    
    """
    :param ndarray X: a matrix 
    :param ndarray omega: a zero-one matrix of the same shape as X 
    :return: a vector of size |omega| of the entries in X that are not zero in omega
    """

    return X[omega != 0]


def binary_weights(Ax, b, number_of_outliers):

    """
    :param ndarray Ax: a vector of length n
    :param ndarray b: a vector of length n 
    :param int number_of_outliers: number of outliers
    :return: a diagonal matrix D with ones and zeros on the diagonal.
             D_(i,i)=1 if the i'th value in abs(Ax - b) is 
             one of the number_of_outliers largest absolute values in abs(Ax-b)  
    """

    residual = Ax - b

    # Get the indices of the p largest values (in absolute values)
    abs_residual = np.abs(residual)
    _, indices = torch.topk(torch.from_numpy(abs_residual), number_of_outliers, largest=True, sorted=False)

    # Convert indices_p from tensor to list for later use
    indices_p = indices.tolist()

    # Create a diagonal matrix D with ones and zeros
    D = identity(len(b), format='csr')
    D[indices_p, indices_p] = 0

    return D
    
def get_U_V_from_solution(x, rank, n1, n2):
    
    VT = x[:rank * n2].reshape(rank, n2, order='F')
    UT = x[rank * n2:].reshape(rank, n1, order='F')
    return UT.T, VT.T
