function [tEnd, out_matrix, stats] = run_test(test, method)


% based on Robust Low-Rank Matrix Completion by Riemannian Optimization
% Léopold Cambier∗, P.-A. Absil†
if method == "RMC" 
   addpath(genpath('Cambier L. and Absil, P.-A. (2016)'))
   I = test.w(:,1);
   J = test.w(:,2);
   A = nonzeros(test.M(sub2ind(size(test.M), test.w(:, 1), test.w(:, 2))));
   
   problem = buildProblemL1(I, J, A, test.m,test.n, test.rank, test.U, test.V');
   problem.lambda = 0;

   params.manopt.maxiter = 40;         
   params.manopt.verbosity = 2;        
   params.manopt.minstepsize = 0;      
   params.manopt.tolgradnorm = 1e-8; 
   params.huber.epsilon = 1;           
   params.huber.theta = 0.05;          
   params.huber.tol = 1e-8;           
   params.huber.itmax = 7;             
   params.huber.verbose = 1;
   tStart = tic;
   [U, V, stats] = rmc(problem, params, true);
   tEnd = toc(tStart);

   out_matrix = U*V;
   rmpath('Robust matrix completion\Cambier L. and Absil, P.-A. (2016)')
end 
% based on Robust-Matrix-Completion-based-on-Factorization-and-Truncated-Quadratic-Loss-Function
% Zhi-Yong Wang, Xiao Peng Li, and Hing Cheung So
if method == "HOAT"
    % as describe by Zhi Yong et al 2023 (Figure 7) ip should be between
    % 2.5 and 4
    ip = 3;
    maxiter = 100;

    tStart = tic;
    [Out_X,U,V,RMSE_1] = HOAT(test.M,test.W,test.rank,maxiter, ip);
    tEnd = toc(tStart);
    out_matrix = Out_X;
    
end

% Accelerated alternating projections for robust principal component analysis
% Hanqin Cai, Jian-Feng Cai, and Ke Wei
if method == "AccAltProj"
    para.trimming = true;
    tStart = tic;
    [L, ~] = AccAltProj(test.M, test.rank, para);
    tEnd = toc(tStart);
    out_matrix = L;    
end

% based on Robust Matrix Factorization by Majorization-Minimization
% Zhouchen Lin, Chen Xu,and Hongbin Zha
if method == "RMF-MM"
     tStart = tic;
    [U,S,V] = svds(test.M,test.rank); 
    U0 = U*sqrt(S);
    V0 = V*sqrt(S);
    para.max_out = 200;
    [U,V] = RMF_MM(test.W,test.M,U0,V0,para);
    tEnd = toc(tStart);
    out_matrix = U*V';
end

%  Practical low-rank matrix approximation under robust l1-norm
%  Y. Zheng, G. Liu, S. Sugimoto, S. Yan, and M. Okutomi.
if method == "RegL1-ALM"
    tStart = tic;
    [out_matrix,~,~,~] = RobustApproximation_M_UV_TraceNormReg(test.M,test.W,test.rank);
    tEnd = toc(tStart);
end 

if method == "LMaFit"
    % tStart = tic;
    % [X,Y,~,~] = lmafit_sms_v1(test.M,test.rank);
    % tEnd = toc(tStart);
    % out_matrix = X*Y;

    problem.m = test.m;
    problem.n = test.n;
    problem.r = test.rank;
    problem.k = length(test.w);
    
    problem.X = test.L(sub2ind(size(test.L), test.w(:,1), test.w(:,2)));
    problem.C = ones(size(problem.X));
    problem.I =  uint32(test.w(:,1));
    problem.J = uint32(test.w(:,2));

    [~, order] = sort(sub2ind([problem.m, problem.n], problem.I, problem.J));
    problem.I = problem.I(order);
    problem.J = problem.J(order);
    problem.X = problem.X(order);
    problem.C = problem.C(order);
    problem.A = test.U;
    problem.B = test.V';
    problem.ind =  sub2ind([problem.m, problem.n], problem.I, problem.J);
    problem.lambda = 1e-8;
    problem.gamma = 10;
    problem.Chat = problem.C.^2 - problem.lambda.^2;
    problem.mask = sparse(double(problem.I), double(problem.J), ...
                          ones(problem.k, 1), problem.m, problem.n, problem.k);
    [U0, sig, V0] = svds(sparse(double(problem.I), double(problem.J), problem.X, problem.m, problem.n, problem.k), problem.r);
    S0 = rand(size(problem.X));
    S0 = rand(size(problem.X));
    clear options;
    options.maxiter = 80;
    options.min_gamma = 1e-8;
    options.epsilon = 30; 
    options.mu1 = 1/10;
    options.mu2 = 1/10;
    options.stepsize = 2;
    
    tStart = tic;
    [X, Y, ~] = LMaFit(problem, U0, V0, options);
    out_matrix = X*Y;
    tEnd =  toc(tStart);
end

if method == "RPCA-GD"
    cd 'Xinyang Yi et al 2016'
    stats = 0;
    params.step_const = 0.5; % step size parameter for gradient descent
    params.max_iter   = 30;  % max number of iterations
    params.tol        = 2e-5;% stop when ||Y-UV'-S||_F/||Y||_F < tol
    tStart = tic;
    [X,Y] = rpca_gd(test.M,test.rank,test.gamma, params);
    tEnd = toc(tStart);
    out_matrix = X*Y';
    cd ..
end

if method == "ManPGmc"
    addpath(genpath('RobustMC_code'))
    problem.m = test.m;
    problem.n = test.n;
    problem.r = test.rank;
    problem.k = length(test.w);
    
    problem.X = test.L(sub2ind(size(test.L), test.w(:,1), test.w(:,2)));
    problem.C = ones(size(problem.X));
    problem.I =  uint32(test.w(:,1));
    problem.J = uint32(test.w(:,2));

    [~, order] = sort(sub2ind([problem.m, problem.n], problem.I, problem.J));
    problem.I = problem.I(order);
    problem.J = problem.J(order);
    problem.X = problem.X(order);
    problem.C = problem.C(order);
    problem.A = test.U;
    problem.B = test.V';
    problem.lambda = 1e-8;
    problem.gamma = 10;
    problem.Chat = problem.C.^2 - problem.lambda.^2;
    problem.mask = sparse(double(problem.I), double(problem.J), ...
                          ones(problem.k, 1), problem.m, problem.n, problem.k);
    [U0, sig, V0] = svds(sparse(double(problem.I), double(problem.J), problem.X, problem.m, problem.n, problem.k), problem.r);
    S0 = rand(size(problem.X));
    clear options;
    options.maxiter = 80;
    options.min_gamma = 1e-8;
    options.epsilon = 30; 
    options.mu1 = 1/10;
    options.mu2 = 1/10;
    options.stepsize = 2;
    tStart = tic;
    [U, V, stats] = ManPGmc(problem,  U0, V0', S0, options);
    tEnd = toc(tStart);
    out_matrix = U*V;
    rmpath 'RobustMC_code'
end

if method == "GNMR"
    clear opts
    opts.verbose = 1;               % display intermediate results
    opts.alpha = 1;                 % variant parameter (e.g., 1: setting, 0: averaging, -1: updating)
    % number of iterations
    opts.max_outer_iter = 50;      % maximal number of outer iterations
    opts.max_inner_iter = 2000;     % maximal number of inner iterations for the LSQR solver
    % early stopping criteria (-1 to disable a criterion)
    opts.stop_relRes = 1e-14;   	% small relRes threshold
                                    % (relRes = ||X_hat - X||_F/||X_hat||_F on the observed entires)
    opts.stop_relDiff = 1e-14;      % small relative X_hat difference threshold
    tStart = tic;
    [out_matrix, ~, ~, ~] = GNMR_completion(test.M,test.w,test.rank, opts);
    tEnd = toc(tStart);
    
end

if method == "RGNMR"
    clear opts
    opts.verbose = 1;               % display intermediate results
    opts.alpha = 1;                 % variant parameter (e.g., 1: setting, 0: averaging, -1: updating)
    % number of iterations
    opts.max_outer_iter = 50;      % maximal number of outer iterations
    opts.max_inner_iter = 1000;     % maximal number of inner iterations for the LSQR solver
    % early stopping criteria (-1 to disable a criterion)
    opts.stop_relRes = 1e-15;   	% small relRes threshold
                                    % (relRes = ||X_hat - X||_F/||X_hat||_F on the observed entires)
    opts.inner_init_tol = 1e-1;
    opts.stop_relDiff = 1e-15;      % small relative X_hat difference threshold
    tStart = tic;
    [out_matrix, ~, ~] = RGNMR(test.M,test.w,test.rank, length(test.s), opts);
    tEnd = toc(tStart);
    
end

if method == "RGNMRLucky"
    clear opts
    opts.verbose = 1;               % display intermediate results
    opts.alpha = 1;                 % variant parameter (e.g., 1: setting, 0: averaging, -1: updating)
    % number of iterations
    opts.max_outer_iter = 50;      % maximal number of outer iterations
    opts.max_inner_iter = 500;     % maximal number of inner iterations for the LSQR solver
    % early stopping criteria (-1 to disable a criterion)
    opts.stop_relRes = 1e-15;   	% small relRes threshold
                                    % (relRes = ||X_hat - X||_F/||X_hat||_F on the observed entires)
    opts.stop_relDiff = 1e-15;      % small relative X_hat difference threshold
    sensitivity = 1;
    tStart = tic;
    [out_matrix, ~, ~, ~] = Robust_GNMR_completion(test.M,test.w,test.rank, floor(length(test.w)*0.1), sensitivity, test, opts);
    tEnd = toc(tStart);
    
end

if method == "AdaptiveRGNMR"
    clear opts
    opts.verbose = 1;               % display intermediate results
    opts.alpha = 1;                 % variant parameter (e.g., 1: setting, 0: averaging, -1: updating)
    % number of iterations
    opts.max_outer_iter = 50;      % maximal number of outer iterations
    opts.max_inner_iter = 2000;     % maximal number of inner iterations for the LSQR solver
    % early stopping criteria (-1 to disable a criterion)
    opts.stop_relRes = 1e-15;   	% small relRes threshold
                                    % (relRes = ||X_hat - X||_F/||X_hat||_F on the observed entires)
    opts.stop_relDiff = 1e-15;      % small relative X_hat difference threshold
    tStart = tic;
    [out_matrix] = adaptive_RGNMR(test.M,test.w,test.rank, test, opts, 0, 0.2, 0.01);
    tEnd = toc(tStart);
    
end

if method == "BSAdaptiveRGNMR"
    clear opts
    opts.verbose = 1;               % display intermediate results
    opts.alpha = 1;                 % variant parameter (e.g., 1: setting, 0: averaging, -1: updating)
    % number of iterations
    opts.max_outer_iter = 50;      % maximal number of outer iterations
    opts.max_inner_iter = 2000;     % maximal number of inner iterations for the LSQR solver
    % early stopping criteria (-1 to disable a criterion)
    opts.stop_relRes = 1e-15;   	% small relRes threshold
                                    % (relRes = ||X_hat - X||_F/||X_hat||_F on the observed entires)
    opts.stop_relDiff = 1e-15;      % small relative X_hat difference threshold
    epsilon = 1e-8;
    tStart = tic;
    [out_matrix, stats.time_1, stats.time_2] = bs_adaptive_RGNMR(test.M,test.w,test.rank, test, opts, 0, 0.5, epsilon);
    tEnd = toc(tStart);
    
end

if method == "AOP"
   addpath(genpath('AOPMCv1\'))
   Known = sort(reshape(sub2ind([test.m test.n], test.w(:,1), test.w(:,2)), [1 length(test.w)])) ;
   data = test.M(Known) ;
   lambda = 1e-8;
   opts1.maxouter = 20;
   opts1.maxinner = 6;
   opts1.gradtol = 1e-5;
   opts1.verbosity = 0;
   opts1.order = 2;
   opts1.computeRMSE = false;
   opts1.U0 = [];
   maxiter = 200;
   tStart = tic;
   [U, V, stats] = BMP_rtrmc(test.m, test.n, test.rank,lambda, Known, data, length(test.s), maxiter, opts1);
   
   tEnd = toc(tStart);

   out_matrix = U*V;
  
   
end 

if method == "BSAdaptiveRTRMC"
   Known = sort(reshape(sub2ind([test.m test.n], test.w(:,1), test.w(:,2)), [1 length(test.w)])) ;
   data = test.M(Known) ;
   lambda = 1e-8;
   opts1.maxouter = 20;
   opts1.maxinner = 6;
   opts1.gradtol = 1e-5;
   opts1.verbosity = 0;
   opts1.order = 2;
   opts1.computeRMSE = false;
   opts1.U0 = [];
   maxiter = 20;
   maxoutiter = 10;
   tStart = tic;
   [U, V, stats] = bs_adaptive_rtrmc(test.m, test.n, test.rank,lambda, Known, data, ...
       ceil(0.2*length(test.w)), 0,maxiter, maxoutiter, opts1);
   tEnd = toc(tStart);

   out_matrix = U*V;
   
end 
