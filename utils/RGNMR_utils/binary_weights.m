function [D] = binary_weights(entriwise_residulas, number_of_outliers)
    % binary_weights returns a diagonal matrix D with ones and zeros on the diagonal.
    % A row has zero if it is one of the p largest rows in abs(a - b).
    %
    % Inputs:
    %   a: a vector
    %   b: a vector 
    %   p: Number of largest rows to consider
    %
    % Output:
    %   D: Diagonal matrix with ones and zeros on the diagonal

    % Get the indices of the p largest rows (in absolute values)
    [~, indices] = sort(entriwise_residulas, 'descend');
    
    indices_p = indices(1:number_of_outliers);

    D = speye(length(entriwise_residulas));
    D(indices_p, indices_p) = 0;
end