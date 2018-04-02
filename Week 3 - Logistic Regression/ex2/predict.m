function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

% Initialize p to be cost for each input vector using the supplied
%   theta values
%
p = sigmoid(X * theta);

% Now iterate over each cost, and update according to the threshold
%
for i = 1:size(p, 1)
    if(p(i) >= 0.5) 
        p(i) = 1;
    else 
        p(i) = 0;
    end
end




% =========================================================================


end
