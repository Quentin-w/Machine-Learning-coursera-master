function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Set up the range of values we'll examine in our CV search
%
p = [.01 .03 .1 .3 1 3 10 30];
len = size(p, 2);

% Create a container to store the results. Each run is stored as a row:
%   C sigma error
%
results = zeros(len ^ 2, 3);
results_idx = 1;

for i = 1:len
    for j = 1:len
       % Capture the C and sigma values for this iteration
       %
       C_i = p(i);
       sigma_j = p(j);
       
       % Train our model using these values
       %
       model= svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_j));
       predictions = svmPredict(model, Xval);
       error = mean(double(predictions ~= yval));
       
       results(results_idx,1) = C_i;
       results(results_idx,2) = sigma_j;
       results(results_idx,3) = error;
       
       results_idx = results_idx + 1;
    end
end

% Examine our results matrix and find the combination with the lowest
%   error
%
[M,I] = min(results);

% Now I(3) will be the index of the row with the smallest error
%
idx = I(3);

C = results(idx, 1);
sigma = results(idx, 2);

% =========================================================================

end
