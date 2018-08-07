function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

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

for i = 1:m
    % set activation units for 1st layer 
    a_layer1 = X(i:i, 1:end);
    
    % compute z_layer2 using theta1
    z_layer2 = Theta1*a_layer1';
    
    % compute g(z_layer2)which is equal to activation units for 2nd layer
    a_layer2 = sigmoid(z_layer2);
    
    % add bias for layer2
    a_layer2 = [1 ; a_layer2];
    
    % compute z_layer3 using theta2
    z_layer3 = Theta2*a_layer2;
    
    % compute g(z_layer3) which is equal to activation units for 3rd layer
    a_layer3 = sigmoid(z_layer3);
    
     % chose label with max probability
    [max_val,label_idx] = max(a_layer3);
    
    % assign the max probaility label index
     p(i,1) = label_idx;
end

% =========================================================================


end
