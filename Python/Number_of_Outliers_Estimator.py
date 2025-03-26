# initialization options
INIT_WITH_SVD = 0
INIT_WITH_RANDOM = 1
INIT_WITH_USER_DEFINED = 2
from RGNMR import *

def number_of_outliers_estimator(X, omega, rank, alpha_min=0, alpha_max=0.5, success_criterion=1e-7, failing_iteration=25, outlier_convergence_criterion=10, verbose=False):
  """
    Run RGNMR algorithm for robust matrix completion
    :param ndarray X: Input matrix (n1,n2). Unobserved entries should be zero
    :param ndarray omega: Mask matrix (n1,n2). 1 on observed entries, 0 on unobserved
    :param int rank: Underlying rank matrix
    :param float alpha_min the minimal fraction of outliers possible, deafult is 0
    :param float alpha_max the maximal fraction of outliers possible, deafult is 0.5
    :param float success_criterion: approximation threshold, if RGNMR estimate reached a relRes smaller than the success_criterion we assume we overestimate the number of outliers
    :param int failing_iteration: the maximal number of iteration to be used when estimating the number of outliers
    :param int outlier_convergence_criterion: if the set of corrupted entries did not change for outlier_convergence_criterion iterations, and relRes did not improve RGNMR stops
    :return: number of outliers
  """
  list_of_number_of_outliers = []
  list_of_errors = []
  list_iterations_since_Lambda_changed = []
  options = {
    # general
    'init_option' : INIT_WITH_SVD,
    'verbose' : False,
    'r_projection_in_iteration' : False,
    # early stopping
    'lsqr_init_tol' : 1e-15,
    # number of iterations
    'max_outer_iter' : failing_iteration,
    'max_inner_iter': 2000,
    # early stopping criteria (-1 to disable a criterion)
    'stop_relRes':  success_criterion,
    'stop_relDiff': 1e-15,
    'stop_Lambda_converged': True,
}
  alpha = alpha_min
  while int((alpha_max - alpha_min)*omega.sum()) > 1:
    list_of_number_of_outliers.append(int(np.ceil(alpha * omega.sum())))
    num_of_outliers = int(np.ceil(alpha * omega.sum()))
    _, _, _, all_relRes, iterations_since_Lambda_changed = RGNMR(X, omega, rank, num_of_outliers, **options)
    list_of_errors.append(all_relRes[-1])
    list_iterations_since_Lambda_changed.append(iterations_since_Lambda_changed)
    if verbose:
      print(f"Number of Outliers {num_of_outliers}, relRes: {all_relRes[-1]}, iterations since Lambda changed: {iterations_since_Lambda_changed}")
    if iterations_since_Lambda_changed ==  0:
      alpha_max = alpha
    else:
      alpha_min = alpha
    alpha = (alpha_max + alpha_min) / 2

  alpha = alpha_max
  return int(np.ceil(alpha * omega.sum())), list_of_number_of_outliers, list_of_errors, list_iterations_since_Lambda_changed