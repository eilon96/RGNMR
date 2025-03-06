function [U, V, L_hat, D] = init_RGNMR(init_option, U_init, V_init, X, omega, rank, outliers_num)
    
    %
    % This method initialize the estimates used by RGNMR
    %%  INPUT
    %   init_option = 0 for SVD, 1 for random, 2 for opts.init_U, opts.init_V
    %   U_init = initial U if init_option is 2
    %   V_init = initial V if init_option is 2
    %   X = the observed matrix 
    %   omega = list of pairs (i,j) of the observed entries
    %   rank = the rank of the target matrix
    %   outliers_num = the expected number of outlires
    %
    %%  OUTPUT
    %   U = initial U
    %   V = initial V
    %   L_hat = initial estimate of the target matrix
    %   D = a diagonal one-zero matrix, represents the initial estimate of
    %   the set of non corrupted entries, see solve_LSQR_problem for
    %   details.

    
    [n1, n2] = size(X);

    %% initialize U and V (of sizes n1 x rank and n2 x rank)
    if init_option == 0
        % initialization by SVD of observed matrix, after applying a
        % threshold operator (remove_top_fraction)
        [U, ~, V] = svds(remove_top_fraction(X, outliers_num/(n1*n2)), rank);
    elseif init_option == 1
        % initialization by random orthogonal matrices
        Z = randn(n1,r);
        [U, ~, ~] = svd(Z,'econ'); 
        Z = randn(n2,r);
        [V, ~, ~] = svd(Z,'econ'); 
    else
        % initazliation by user-defined matrices
        U = U_init;
        V = V_init; 
    end
    % construct initial estimate
    L_hat = U * V';

    % vectorize the input matrix X
    vector_X = vectorize_observed_matrix(X, omega);
    % vectorize the initial estimator
    vector_L_hat = vectorize_observed_matrix(L_hat, omega);
    % construct D
    D = binary_weights(abs(vector_L_hat - vector_X), outliers_num);
end



function X = remove_top_fraction(X, alpha)
    %
    %   This function is a hard thresholding operator, used at initialization of
    %   RGNMR. The function removes the oversized elements in the observed
    %   matrix X. 
    %
    %%  INPUT 
    %   X = the observed matrix
    %   alpha = the fraction of entries to remove from each row and column
    %   
    %%  OUTPUT
    %   X = the observed matrix after the thresholding operator 
    %   

    [m, n] = size(X);
    num_row = max(1, round(alpha * n)); % Number of elements to remove per row
    num_col = max(1, round(alpha * m)); % Number of elements to remove per column
    
    % apply thresholding to each row
    for i = 1:m
        [~, idx] = maxk(abs(X(i, :)), num_row); % find indices of top values in row
        X(i, idx) = 0; % set them to zero
    end
    
    % apply thresholding to each column
    for j = 1:n
        [~, idx] = maxk(abs(X(:, j)), num_col); % find indices of top values in column
        X(idx, j) = 0; % set them to zero
    end
end