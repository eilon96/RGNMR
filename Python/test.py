import generate_data
from Number_of_Outliers_Estimator import *
from RGNMR_utils import *
from RGNMR import *
import time
from tqdm import tqdm

def run_test():
    """
    Run a single simulation.
    First estimates the number of corrupted entries then uses this nuber as an upper bound of k_star
    to recover the true matrix using RGNMR.
    """
    # matrix parameters
    n1 = 800
    n2 = 800
    rank = 5
    oversampling_ratio = 8
    fraction_of_outliers = 0.05
    condition_number = 2

    # generate X and omega
    X, L_star, omega, num_of_outliers = generate_data.generate_test(n1, n2, rank, condition_number, oversampling_ratio,
                                                               fraction_of_outliers)
    # estimate number of outliers
    print("\nEstimate number of outliers\n\n")
    estimated_number_of_outliers = number_of_outliers_estimator(X, omega, rank, verbose=True, alpha_min=0, alpha_max=0.5, failing_iteration=50)

    # options for RGNMR use
    options = {
        # general
        'r_projection_in_iteration' : False,
        'lsqr_init_tol' : 1e-15,
        'max_outer_iter' : 50,
        'max_inner_iter': 2000,
        'stop_relRes':  1e-14,
        'stop_relDiff': 1e-14,
    'init_option' : INIT_WITH_SVD

    }
    # recover the matrix using RGNMR
    print("\nRecover matrix\n\n")
    L_hat,iter, convergence_flag, all_relRes, iterations_since_Lambda_changed = RGNMR(X, omega, rank, estimated_number_of_outliers, **options)
    #compute relative error
    true_error = np.linalg.norm(L_hat - L_star, ord='fro') / np.linalg.norm(L_star, ord='fro')
    print(f"\nEstimated number of outliers: {estimated_number_of_outliers}, True Number of Outliers: {num_of_outliers}\n")
    print(f'relative error: {true_error}\n')

if __name__ == '__main__':

    run_test()
