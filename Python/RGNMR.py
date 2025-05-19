import numpy as np
from scipy import sparse
from scipy.sparse import linalg, identity
from sklearn.preprocessing import normalize
import gc
from RGNMR_utils import *

# initialization options

def RGNMR(X, omega, rank, num_of_outliers,
                     verbose=True, show_matrix=False,
                     max_outer_iter=100, max_inner_iter=2000,
                     lsqr_init_tol=1e-1, lsqr_smart_tol=False,
                     init_option=INIT_WITH_SVD, init_U=None, init_V=None,
                     stop_relRes=1e-16, stop_relDiff = -1, stop_relResDiff = -1,stop_Lambda_converged=False,
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

    # set the initial estimates  U, V, X_hat, D
    U, V, L_hat, D = init_RGNMR(init_option, X, omega, rank, num_of_outliers, init_U, init_V)


    # generate sparse indices to accelerate future operations
    sparse_matrix_rows, sparse_matrix_columns = generate_sparse_matrix_entries(omega, rank)

    # before iterations
    early_stopping_flag = False
    current_tol = lsqr_init_tol
    all_relRes = [1]
    iter_num = 0
    iterations_since_Lambda_changed = 0
    restart = True

    # iterations
    while iter_num < max_outer_iter and not early_stopping_flag:

        iter_num += 1
        L_hat_previous = L_hat
        D_previous = D

        # solve the least of squares problem
        U, V, L_hat, entriwise_residuals, relRes = solve_LSQR_problem(X, U, V, omega, D, sparse_matrix_rows, sparse_matrix_columns, current_tol, max_inner_iter)

        # estimate the set of non corupted entries
        D = binary_weights(entriwise_residuals, num_of_outliers)

        if relRes > 1e-5 and iter_num > 50 and restart:
            v = D*vectorize_observed_matrix(X, omega)
            X_tag = np.zeros_like(X)
            X_tag[omega==1] = v
            U, V, L_hat, D = init_RGNMR(init_option, X_tag, omega, rank, num_of_outliers, init_U, init_V)
            restart = False

        iterations_since_Lambda_changed = (iterations_since_Lambda_changed + 1) * ((D != D_previous).nnz == 0)
        # decrease the tolorence for error
        if lsqr_smart_tol:
          current_tol =  current_tol*1e-1

        # report RGNMR progression
        report_RGNMR_progression(verbose, iter_num, relRes)
        all_relRes.append(relRes)

        # check early stopping criteria
        early_stopping_flag = check_early_stopping_criteria(early_stopping_flag, relRes, stop_relRes, all_relRes, stop_relDiff, L_hat, L_hat_previous,
                                                            stop_Lambda_converged, iterations_since_Lambda_changed, stop_relResDiff, verbose)



    # return
    convergence_flag = iter_num < max_outer_iter
    (U_r, Sigma_r, V_r) = linalg.svds(L_hat, k=rank, tol=1e-17)
    L_hat = U_r @ np.diag(Sigma_r) @ V_r

    return L_hat, iter_num, convergence_flag, all_relRes, iterations_since_Lambda_changed
