function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Compute z just to make the code simpler below
%
z = X * theta;

% Now the cost is just the function but using element-multiplication
%   inside the sum()
%
J = (-1 / m) * sum(y .* log(sigmoid(z)) + (1 - y) .* (log(1 - sigmoid(z))));

% Now the partial derivative is w.r.t. a single term in theta, so to 
%   compute it we'll iterate over the values of

% temp will be a vector that will hold the next theta values while
%   we calculate them
%
tempTheta = zeros(size(X, 2), 1);

for i = 1:size(X, 2)
    tempTheta(i) = (1 / m) * sum((sigmoid(z) - y) .* X(:,i));
end

grad = tempTheta;

% =============================================================

end
