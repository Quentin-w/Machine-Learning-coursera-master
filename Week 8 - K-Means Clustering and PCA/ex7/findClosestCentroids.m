function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);
m = size(X, 1);

% You need to return the following variables correctly.
idx = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% Loop over every vector in X to determine its assignment
%
for i = 1:m
    % Create a vector containing the distance between x_i and each
    %   centroid
    %
    x_i = X(i,:);
    idx_i = zeros(1, K);
    
    % Iterate over each centroid to calculate its distance
    %
    for j = 1:K
        % Pull out the j'th centroid
        %
        c_j = centroids(j,:);
        
        % Calculate the distance between x_i and this c_j centroid
        %
        idx_i(1, j) = sum((x_i - c_j) .^ 2); 
    end
    
    % Now extract the index of the minimum distance captured in idx_i
    %
    [M, I] = min(idx_i);
    
    idx(i) = I(1);
end


% =============================================================

end

