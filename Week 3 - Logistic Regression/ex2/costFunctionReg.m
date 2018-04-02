function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


% Compute z just to make the code simpler below
%
z = X * theta;

% The regularization term is the squared sum of all theta values
%   times the lambda coefficient...but excluding theta(1)!
%
reg_term = (lambda / (2 * m)) * (sum(theta .^ 2) - theta(1)^2);

% Now the cost is just the function but using element-multiplication
%   inside the sum()
%
J = (-1 / m) * sum(y .* log(sigmoid(z)) + (1 - y) .* (log(1 - sigmoid(z)))) + reg_term;

% Now the partial derivative is w.r.t. a single term in theta, so to 
%   compute it we'll iterate over the values of

% temp will be a vector that will hold the next theta values while
%   we calculate them
%
tempTheta = zeros(size(X, 2), 1);

% Initialize the regularization term for the partial derivatives
%   to zero, so it handles the case of i = 1 properly
%
reg = 0;

for i = 1:size(X, 2)
    % Compute the regularization term, but handle the special 
    %   case were the term for theta(1) should be zero
    %
    reg(i>1) = (lambda / m) * theta(i);
    
    tempTheta(i) = (1 / m) * sum((sigmoid(z) - y) .* X(:,i)) + reg;
end

grad = tempTheta;


% =============================================================

end
