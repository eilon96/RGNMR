import numpy as np
from scipy import sparse
from scipy.sparse import linalg, identity
from sklearn.preprocessing import normalize
import gc
from RGNMR_utils import *

# initialization options
INIT_WITH_SVD = 0
INIT_WITH_RANDOM = 1
INIT_WITH_USER_DEFINED = 2
MAX_SIZE_FOR_VISUALIZATON_OF_ESTIMATED_MATRIX = 20 

def RGNMR_completion(X, omega, rank, num_of_outliers, 
                     verbose=True, show_matrix=False,
                     max_outer_iter=100, max_inner_iter=2000, 
                     lsqr_init_tol=1e-1, lsqr_smart_tol=True,
                     init_option=INIT_WITH_SVD, init_U=None, init_V=None,
                     stop_relRes=1e-16, stop_relDiff = -1, stop_relResDiff = -1, 
                     r_projection_in_iteration=False, return_a_list_of_estimators=False):
    """
    Run RGNMR algorithm for robust matrix completion
    :param ndarray X: Input matrix (n1,n2). Unobserved entries should be zero
    :param ndarray omega: Mask matrix (n1,n2). 1 on observed entries, 0 on unobserved
    :param int rank: Underlying rank matrix
    :param bool verbose: if True, display intermediate results
    :param bool show_matrix: if True, display intermediate estimates as latex, works only with IPython
    :param int max_outer_iter: Maximal number of outer iterations
    :param int max_inner_iter: Maximal number of inner iterations
    :param float lsqr_init_tol: initial tolerance of the LSQR solver
    :param bool lsqr_smart_tol: if true the lsqr tolorence decreases at each iteration by a factor of 1e-1 
    :param int init_option: how to initialize U and V (INIT_WITH_SVD, INIT_WITH_RAND, or INIT_WITH_USER_DEFINED)
    :param ndarray init_U: U initialization (n1,rank), used in case init_option==INIT_WITH_USER_DEFINED
    :param ndarray init_V: V initialization (n2,rank), used in case init_option==INIT_WITH_USER_DEFINED
    :param float stop_relRes: relRes threshold for ealy stopping (relevant to noise-free case), -1 to disable
    :param float stop_relDiff: relative X_hat difference threshold for ealy stopping, -1 to disable
    :param float stop_relResDiff: relRes difference difference threshold for early stopping, -1 to disable
    :parma bool r_projection_in_iteration: if true, error estimation at each iteration
      is calculated for the best rank-r approximation using SVD
    :return: GNMR's estimate, final iteration number, convergence flag and all relRes
    """


    n1, n2 = X.shape
    num_visible_entries = np.count_nonzero(omega)
    visible_ratio = num_visible_entries / (n1*n2)
    
    # in order to dispaly the matrix in_Python needs to be True
    in_IPython = 'get_ipython' in globals()

    # set the initial estimate
    if init_option == INIT_WITH_SVD:
      # applies a thresholding operator on X then applies svd 
      (U, _, V) = linalg.svds(threshold_operator(torch.from_numpy(X), omega, num_of_outliers/omega.sum()).numpy(), k=rank, tol=1e-16)
      V = V.T
    elif init_option == INIT_WITH_RANDOM:
      U = np.random.randn(n1, rank)
      V = np.random.randn(n2, rank)
      U = np.linalg.qr(U)[0]
      V = np.linalg.qr(V)[0]
    else:
      U = init_U
      V = init_V

    # generate sparse indices to accelerate future operations
    sparse_matrix_rows, sparse_matrix_columns = generate_sparse_matrix_entries(omega, rank, n1, n2)

    # generate (constant) b for the least squares problem
    b = generate_b(X, omega)

    # before iterations
    early_stopping_flag = False
    relRes = 1
    current_tol = lsqr_init_tol
    all_relRes = [relRes]
    all_estimators = []
    best_relRes = np.max(np.abs(X))
    X_hat = U @ V.T        # intial estimate
    X_hat_best_2r = X_hat  # stores best intermediate rank 2r estimate

    # choose the set of non corrupted entries based on the initial estimate
    vectorize_X_hat = generate_b(X_hat, omega)
    D = binary_weights(vectorize_X_hat, b, num_of_outliers)

    # iterations
    iter_num = 0
    while iter_num < max_outer_iter and not early_stopping_flag:
        iter_num += 1

        ## build the least of squares problem
        # A is a sparse matrix of shape (|Omega|,(n1+n2)*rank)
        A = generate_sparse_A(U, V, omega, sparse_matrix_rows, sparse_matrix_columns, num_visible_entries, n1, n2, rank)
        # b is a vecorization of the non zero entries in omega*(X  + U@V.T) with shape (|omega|,)  
        b = generate_b(X + U @ V.T, omega)

        # solve the least squares problem, x is the solution , res = ||DAx - Db||
        x, _, _, res = linalg.lsqr(D@A, D@b, atol=current_tol, btol=current_tol, iter_lim=max_inner_iter)[:4]

        # estimate the set of non corupted entries
        D = binary_weights(A@x, b, num_of_outliers)
        
        # A can get very large, we therefore free memory by deleting it 
        del A
        gc.collect()
        
        # obtain new estimates for U and V
        # x = convert_x_representation(x, rank, n1, n2)
        U_next, V_next = get_U_V_from_solution(x, rank, n1, n2)

        # get new estimate and calculate corresponding error
        relRes = res / np.linalg.norm(D@b)
        X_hat_previous = X_hat
        X_hat_2r = U @ V_next.T + U_next @ V.T - U @ V.T
        if r_projection_in_iteration:
          (U_r, Sigma_r, V_r) = linalg.svds(X_hat_2r, k=rank, tol=1e-17)
          X_hat = U_r @ np.diag(Sigma_r) @ V_r
        else:
          X_hat = U_next @ V_next.T
        X_hat_diff =  np.linalg.norm(X_hat - X_hat_previous, ord='fro') / np.linalg.norm(X_hat, ord='fro')

        all_relRes.append(relRes)
        if relRes < best_relRes:
          best_relRes = relRes
          X_hat_best_2r = X_hat_2r


        # update U, V
        U = U_next
        V = V_next

        # report
        if show_matrix and in_IPython and max(n1, n2) < MAX_SIZE_FOR_VISUALIZATON_OF_ESTIMATED_MATRIX:
          time.sleep(2)
          display.clear_output(wait=True)
          (U_r, Sigma_r, V_r) = linalg.svds(X_hat_2r, k=rank, tol=1e-17)
          current_estimate =  U_r @ np.diag(Sigma_r) @ V_r
          print_latex(f"L_{{{iter_num}}} = " + matrix_to_latex(current_estimate, np.zeros_like(X_hat), np.ones_like(X_hat)))
        if verbose:
          print("[INSIDE GNMR] iter: " + str(iter_num) + ", relRes: " + str(relRes))

        # we search for a more accurate solution at each iteration by decreasing the tolorence for error
        if lsqr_smart_tol:
          current_tol =  current_tol*1e-1

        # check early stopping criteria
        if stop_relRes > 0:
          early_stopping_flag |= relRes < stop_relRes
        if stop_relDiff > 0:
          early_stopping_flag |= X_hat_diff < stop_relDiff
        if stop_relResDiff > 0:
          early_stopping_flag |= np.abs(relRes / all_relRes[-2] - 1) < stop_relResDiff
        if verbose and early_stopping_flag:
          print("[INSIDE GNMR] early stopping")


    # return
    convergence_flag = iter_num < max_outer_iter
    (U_r, Sigma_r, V_r) = linalg.svds(X_hat_best_2r, k=rank, tol=1e-17)
    X_hat = U_r @ np.diag(Sigma_r) @ V_r

    return X_hat, iter_num, convergence_flag, all_relRes