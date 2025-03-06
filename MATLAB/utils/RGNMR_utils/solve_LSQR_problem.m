function [U, V, L_hat, entriwise_residulas, relRes, LSQR_iters_done] = ...
         solve_LSQR_problem(X, U, V,omega, D, colind_A, tol, max_iter)
  %
  %  At each iteration RGNMR solves
  %             (U_next, V_next) = min_{U^, V^} ||U*V^' + U*V^' - U*V' - X||_{F(\Omega \cap \Lambda_{t})}
  %  We construct this problem as a weighted least of squares problem: min_x ||DAx-Db||.
  %  The matrix D define binary weights, with zeros on the entries that are
  %  estimated to be corrupted, and ones otherwise.
  %% INPUT:
  %  X = observed matrix
  %  U = current U estimate,  shape (n1,rank)
  %  V = current V estimate,  shape (n2,rank)
  %  omega = list of pairs (i,j) of the observed entries
  %  D = a one-zero diagonal matrix (details below)
  %  colind_A = used to build the lsqr problem
  %  tol = the tolorence of the lsqr solver
  %  max_iter = maximal number of iteration to be used by the lsqr solver
  %
  %% OUTPUT
  %  U_next = next U estimate
  %  V_next = next V estimate
  %  L_hat  = next L estimate
  %  entriwise_residual = the residuals of the new estimate L_hat from the observed matrix
  %  relRes = the relative error of the new estimate
  %  LSQR_iters_done = number of iterations used by the lsqr solver
  % 
    
    [n1, rank] = size(U);
    [n2, ~] = size(V);

    %% construct variables A,b for LSQR to solve |DAx-Db|^2 
    % construct A
    A = generate_sparse_A(U, V, omega, colind_A);    
    % construct b 
    X_updated = X + U*V';
    b = vectorize_observed_matrix(X_updated, omega);

    %% solve the least squares problem
    [x, ~, relRes, LSQR_iters_done] = lsqr(D*A, D*b, tol, max_iter);

    %% construct U_next and V_next from the solution x
    U_next = zeros(size(U));
    V_next = zeros(size(V)); 
    nc_list = rank * (0:1:(n2-1)); 
    for i=1:rank
        V_next(:,i) = x(i+nc_list); 
    end
    nr_list = rank * (0:1:(n1-1)); 
    start_idx = rank*n2; 
    for i=1:rank
        U_next(:,i) = x(start_idx + i + nr_list);
    end
    L_hat = U * V_next' + U_next * V' - U * V';
    U = U_next;
    V = V_next;
    entriwise_residulas = abs(A*x - b);
end

%% auxiliary function for generating sparse matrix A
function A = generate_sparse_A(U,V,omega,colind_A)
    nv = size(omega,1);
    r = size(U,2);
    n1 = size(U,1);
    n2 = size(V,1);
    %A = zeros(nv,m);   % matrix of least squares problem 
    val_A = zeros(1,2*r*nv);
    rowind_A=kron(1:nv,ones(1,2*r));
    val_A = generate_val_A(U, V, omega, colind_A);
    %{
    for counter=1:nv
        j = omega(counter,1); k = omega(counter,2);
        %colind_A = [colind_A,(r*(k-1)+1):(r*(k-1)+r),(r*nc + r*(j-1)+1):(r*nc + r*(j-1)+r)];
        val_A((counter-1)*(2*r)+1:counter*2*r) = [U(j,:),V(k,:)];
    end
    %}
    A = sparse(rowind_A,colind_A,val_A,nv,r*(n1+n2));
end
