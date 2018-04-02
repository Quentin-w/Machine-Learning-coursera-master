function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix to include the bias node
%
X = [ones(m, 1) X];

% Calculate the a2 layer for all input rows
%   After this the a1 values for the first row of X is now
%   the first column of a2
%
a2 = sigmoid(Theta1 * X');

% Now add in the bias node for each a1 layer by prepending
%   a 1's row vector to a2
%
a2 = [ones(1, size(a2, 2)); a2];

% Now we calculate the a3 layer using the updated a2 matrix
%
a3 = sigmoid(Theta2 * a2);
    
% Now a3 contains a column vector of probabilities for each of
%   possible classifications. So if we started with 10 examples,
%   and 5 possible classes, a3 will be a 5x10 matrix
%
% With this we want to extract the row for each column that contains
%   the maximum probability
%
[M, p] = max(a3);

% Now p is a row-vector with the index of the largest probability in each
%   column, but we need to return a column vector, so just transpose it
%
p = p';

% =========================================================================

end
