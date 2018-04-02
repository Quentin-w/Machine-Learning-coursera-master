function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Feed forward - Compute the 'a' vectors using the provided data
%

% Convert the label vector (y) to a matrix with 1's
%
Y = zeros(length(y),num_labels);
for i = 1:m
   label = y(i);
   Y(i, label) = 1;
end

%% Feed-forward propagation
%
% Add ones to the X data matrix to include the bias node (which 
%   is a1)
%
a1 = [ones(m, 1) X];

% Calculate the a2 layer for all input rows
%   After this the a1 values for the first row of X is now
%   the first column of a2
%
z2 = Theta1 * a1';
a2 = sigmoid(z2);

% Now add in the bias node for each a1 layer by prepending
%   a 1's row vector to a2
%
a2 = [ones(1, size(a2, 2)); a2];

% Now we calculate the a3 layer using the updated a2 matrix
%
z3 = Theta2 * a2;
a3 = sigmoid(z3);



%% Compute the cost of the network after FFP
%
% Now a3 contains a column vector of probabilities for each of
%   possible classifications. So if we started with 10 examples,
%   and 5 possible classes, a3 will be a 5x10 matrix
%
% Now that we have the output nodes, we can use them to calculate
%   the error of the network. To do so we'll iterate over every
%   
error = 0;

% The error for the network is based on every sample we have, so
%   iterate over all the available samples
%
for i = 1:m
   % Compute the error for this specific sample by examining how far
   %    off the output value is for each possible label. So we need to
   %    iterate over all possible labels and compute the error for each
   %   
   for j = 1:num_labels
       % Just grab the actual label for the j'th node
       %
       y_j = Y(i, j);
       
       % Grab the predicted value for the j'th node. Note that because 
       %    of our vectorized implementation above, a3 is oriented the 
       %    other way so we swap the indexes
       %
       a_j = a3(j, i);
       
       % Now the error at this output node (the j'th) is calculated
       %    using the predicted value and the actual. So we'll have
       %    some decimal value (from a3) and will compare it with
       %    the 0 or 1 value that exists in the Y matrix we created
       %    above.
       %
       error_j = (-y_j * log(a_j)) - (1 - y_j) * (log(1 - a_j));
       
       % Now that we have the error for the j'th node of the i'th
       %    training sample, we just add it to the running total
       %    for the entire network
       %
       error = error + error_j;
   end
end

% Finally, now that we have the sum'd error for all samples in the data
%   set we take the average by dividing it by the number of samples
%
J = error / m;

% Now we need to add in the regularization using the theta matrices
%   provided, but we use temp matrices so we can set the first column
%   (the bias nodes) to zero first
%
Theta1_reg = Theta1;
Theta1_reg(:,1) = zeros(size(Theta1, 1), 1);

Theta2_reg = Theta2;
Theta2_reg(:,1) = zeros(size(Theta2, 1), 1);

% Compute the element-wise square of each term in these copies of
%   the theta matrices
%
r2 = sum(Theta1_reg .^ 2);
r3 = sum(Theta2_reg .^ 2);

% Finally update the overall cost using these regularized terms
%
J = J + (lambda / (2 * m)) * (sum(r2) + sum(r3));


%% Vectorized Backpropagation using our calculated output values

% We can vectorize the initial calculation of delta_3. In this
%   example, a3 is 10x5000 and Y is the logical matrix with size
%   5000x10. So in the vector operation we use Y'.
%
% The result of this is a 10x5000 matrix where each column measures
%   how far off each output node was for that specific input.
%
Delta3 = a3 - Y';

% Now calculate Delta2 using Delta3. Note we remove the first column
%   of Theta2 here because we don't include the bias nodes in this
%   specific calculation
%
Theta2_temp = Theta2(:, 2:end);
Delta2 = (Theta2_temp' * Delta3) .* sigmoidGradient(z2);


% Now the gradient is the measure of how much each element in
%   Theta2 contributed to the error seen in a3. We calculate this
%   across all the samples using a vector operation using the error
%   (Delta3) and the input values (a2). This multiplication will
%   sum the errors for each sample, so we divide by the sample size
%   to get the average error.
% 
% We'll start this operation by computing the regularization term
%   with Theta2, but with the bias node values set to zero
%
Theta2_reg = Theta2;
Theta2_reg(:,1) = zeros(1, size(Theta2_reg, 1));
Theta2_grad = (Delta3 * a2' + (lambda * Theta2_reg)) / m;

% Since this network only has a single hidden layer, we don't have
%   another "Delta" to compute. To calculate the gradient for Theta1
%   we can just use the Delta2 matrix we just created. This is the 
%   same calc as above, and helps determine how much the weight of
%   each input node contributed to the error seen at the hidden layer
%
Theta1_reg = Theta1;
Theta1_reg(:,1) = zeros(1, size(Theta1_reg, 1));
Theta1_grad = (Delta2 * a1 + (lambda * Theta1_reg)) / m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
