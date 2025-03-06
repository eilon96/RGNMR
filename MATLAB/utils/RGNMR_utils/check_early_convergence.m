function [convergence_flag] = check_early_convergence(relRes, all_relRes, stop_relRes, ...
                                                L_hat, L_hat_previous,  stop_relDiff, ...
                                                stop_relResDiff, number_of_iterations_since_D_changed, stop_D_diff, ...
                                                iter, LSQR_iters_done, verbose)
    %
    %   The function checks if some condition were met by the current
    %   estimate L_hat. If so it indicates that the RGNMR converged 
    %
    %%  INPUT
    %   relRes = residual errors of the least-squares problem
    %   all_relRes = list of the residual errors of the least-squares problem throughout the iterations
    %   stop_relRes = if relRes < stop_relRes sets convergence_flag to true
    %   L_hat = current estimate of L
    %   L_hat_previous = previous estimate of L
    %   stop_relDiff = if the differnce between L_hat and L_hat previous is smaller than stop_relDiff sets convergence_flag to true
    %   stop_relResDiff = if abs(all_relRes(iter-1)/relRes-1) < stop_relResDiff sets convergence_flag to true
    %   iter = number of iteration of RGNMr
    %   LSQR_iters_done = number of iteration used by the lsqr solver
    %   verbose = if true display the reason of convergence
    %
    %%  OUTPUT
    %   convergence_flag = indicates whether RGNMR converged

    convergence_flag = 0;
    L_hat_diff = norm(L_hat - L_hat_previous, 'fro') / norm(L_hat, 'fro');
    if relRes < stop_relRes
        msg = '[INSIDE GNMR] Early stopping: small error on observed entries\n';
        convergence_flag = 1;    
    elseif L_hat_diff < stop_relDiff
        msg = '[INSIDE GNMR] Early stopping: L_hat does not change\n';
        convergence_flag = 1;
    elseif iter > 1 && ...
            abs(all_relRes(iter-1)/relRes-1) < stop_relResDiff
        msg = '[INSIDE GNMR] Early stopping: relRes does not change\n';
        convergence_flag = 1;   
    elseif iter > 1 && LSQR_iters_done == 0
        msg = '[INSIDE GNMR] Early stopping: no iterations of LSQR solver\n';
        convergence_flag = 1;
    elseif number_of_iterations_since_D_changed > stop_D_diff && ...
                                    mod(log10(all_relRes(iter - 1)), 10) - mod(log10(relRes), 10) < 1
        msg = '[INSIDE GNMR] Early stopping: no change on the set of outliers, and no imporvent in estimate\n';
        convergence_flag = 1;
    end

    if convergence_flag
        if verbose
            fprintf(msg);
        end
    end
end