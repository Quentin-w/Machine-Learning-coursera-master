function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
%g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

% We can use the element-wise operation here to seamlessly calculate
%   the sigmoid of any scalar, vector or matrix input
%
g = 1 ./ (1 + exp(-z))



% =============================================================

end
