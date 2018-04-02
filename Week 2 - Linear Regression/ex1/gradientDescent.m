function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    % We use temp variables because these updates need to be done
    %   simultaneously.
    %
    % Computing temp1 uses the first value of theta and is simpler
    %   because we don't need to consider the first column of X (which
    %   is all 1's because of how we set up the problem
    %
    temp1 = theta(1) - (alpha / m) * sum((X * theta) - y); 
    
    % The second value of theta is a little more complicated to calc
    %   but we basically do the same as the first value, but we need
    %   to also consider the second column of X for each term. As we
    %   compute the i'th term in the summation we need to multiply
    %   each term by the corresponding term in the second column of
    %   X.
    %
    temp2 = theta(2) - (alpha / m) * sum(((X * theta) - y) .* X(:,2));

    theta = [temp1; temp2];




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
