function [M, U, V, D, mu] = generate_matrix(m, n, rank, condition_number)
%% Input 
% m : number of rows
% n : number of columns
% rank: the rank of the matrix toi generate
% condition_number : the condition  number of the matrix

%% Output
% M = UDV it's SVD decomposition
% mu : the incohernece of the matrix 

singluar_values = linspace(1, condition_number, rank);
D = diag(singluar_values);

Z = randn(m,rank); 
[U, ~, ~] = svd(Z,'econ'); 


Z = randn(n,rank); 
[V, ~, ~] = svd(Z,'econ'); 


M = U * D * V';

% find the incohernece 
mu_U = max(sum(U.^2,2)) * m / rank;
mu_V = max(sum(V.^2,2)) * n / rank;

mu = sqrt(max(mu_U, mu_V));

end