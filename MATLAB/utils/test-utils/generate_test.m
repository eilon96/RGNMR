function [test,flag] = generate_test(m, n, rank, condition_number, ...
    p, gamma, mu_1, sigma_1)

%% Input 
% m : number of rows
% n : number of columns
% rank: the rank of the matrix toi generate
% condition_number : the condition  number of the matrix

% p : over_sampling_ratio
% gamma : frequency of outliers

% mu_1,sigma_1: the expected value and variance of the
% gaussian that defines the noise

%% Output
% test.M : the ovbservable matrix
% test.m : number of rows in the matrix
% test.m : number of columns in the matrix
% test.m : rank of the matrix
% test.L : the true matrix
% test.incoh :  Incoherence of L
% test.N_1 : noise matrix
% test.N_2 : outliers matrix
% test.W : the mask 
% test.w : indices of observable entries (2d array)
% test.S : outliers mask
% test.s : indices of outliers entries(2d array)
% test.mu_1/sigma_1: the expected value and variance of the
% gaussian that defines the noise

% test.mu_2/sigma_2: the expected value and variance of the
% gaussian that defines the outliers

test.m = m;
test.n = n;
test.condition_number = condition_number;
test.oversampling = p;
test.gamma = gamma;
test.rank = rank;
test.mu_1  = mu_1;
test.sigma_1 = sigma_1;

% the original matrix 
[test.L, test.U, test.V, test.D, test.incoh] = generate_matrix(m, n, rank, condition_number);

% noise matrix
test.N_1 = normrnd(mu_1,sigma_1, m, n);

% mask
k =  min(floor(rank*(m+n-rank)* p), m*n);
%k = floor(p*m*n);
[test.W,test.w, flag]  = generate_mask(m,n ,k, rank, 10000);
if flag == 0
    disp('FAILED TO GENERATE TEST');
    return
end

% create outlier matrix N_2

N_2 = unifrnd((-1)*max(abs(test.L), [], 'all'), max(abs(test.L), [], 'all'), m, n);
[test.S, test.s, flag] = generate_outliers(test.W, test.w, gamma, test.rank);
    if flag == 0
        test.N_2 = N_2 .* test.S;
    else
        disp('FAILED TO GENERATE TEST');
        return
    end


% create the observable matrix
test.M = test.W .*(test.L + test.N_1 + test.N_2);



