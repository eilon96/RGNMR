function [H, omega_2d, reject] = generate_outliers(W, w, gamma, rank)
    %% Input
    % W : (mxn) sparse matrix 
    %     ones at chosen  entries and
    %     zeros at unchosen  entries.
    % w : all indices of chosen enries 
    % gamma : probabilty to choose a an outlier

    %% Output
    reject = 1;
    reject_number = 0;
    reject_max = 10;
    m = size(W,1);
    n = size(W, 2);
    while(reject == 1)
        omega = (sort(randperm(size(w, 1),floor(gamma*size(w, 1)))))';
        omega_2d = w(omega,:);
        i_Omega = omega_2d(:,1);
        j_Omega = omega_2d(:,2);
        H = sparse(i_Omega,j_Omega,ones(floor(gamma*size(w, 1)),1),m,n);
        nr_entr_col_W_H = sum(W-H,1)';
        nr_entr_row_W_H = sum(W-H,2);

        if (isempty(find(nr_entr_col_W_H<rank,1)) == 1) && (isempty(find(nr_entr_row_W_H<rank,1)) == 1)
            reject = 0;
        else
            reject_number = reject_number + 1;
            if reject_number > reject_max
                 disp('No mask found!');
                 break
            end 
        end 
    end
end