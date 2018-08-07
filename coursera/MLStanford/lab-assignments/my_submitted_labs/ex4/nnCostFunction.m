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

% Add ones to the X data matrix
X = [ones(m, 1) X];

total_sum_m = 0;

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
    
    % recode labels for y as a 10 dim vector
    y_recoded = zeros(10,1);
    y_recoded(y(i),1) = 1;
    
    total_sum_k = 0;
    for k = 1:num_labels
        sum_k = (-y_recoded(k)*log(a_layer3(k))) - ((1-y_recoded(k))*log(1-a_layer3(k)));
        total_sum_k = total_sum_k + sum_k;
    end
    
    total_sum_m = total_sum_m + total_sum_k;
    
     % chose label with max probability
    %[max_val,label_idx] = max(a_layer3);
    
    % assign the max probaility label index
     %p(i,1) = label_idx;
end

% compute cost
J = total_sum_m/m;

% compute regularized cost function
reg_sum1 = 0; % first component of regularized sum
reg_sum2 = 0; % second component of regularized sum

% compute 1st component of regularized sum taking Theta1 values
for j = 1:hidden_layer_size
    sum = 0;
    for k = 2:input_layer_size+1
        sum = sum + Theta1(j,k) * Theta1(j,k);
    end
    reg_sum1 = reg_sum1 + sum;
end

% compute 2nd component of regularized sum taking Theta2 values
for j = 1:num_labels
    sum = 0;
    for k = 2:hidden_layer_size+1
        sum = sum + Theta2(j,k) * Theta2(j,k);
    end
    reg_sum2 = reg_sum2 + sum;
end

% total regularized sum
total_reg_sum = (lambda/(2*m))*(reg_sum1 + reg_sum2);

% compute cost with reqularized terms
J = J + total_reg_sum;

% -------------------------------------------------------------

% Part 2
% ==============
% iterate through each training example
Delta1 = 0;
Delta2 = 0;

for t = 1:m
    % step 1: 
    % --------
    % set activation units for 1st layer 
    a_layer1 = X(t:t, 1:end);
    
    %add bias for layer1
    %a_layer1 = [1; a_layer1];
    
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
    
    % step 2: for each output unit k in layer 3 (the output layer),
    % calculate delta(k)
    
    d3 = zeros(num_labels,1);
    
    y_k = 0;
    
    for k = 1:num_labels
        if y(t) == k
            y_k = 1;
        else
            y_k = 0;
        end
        d3(k,1) = a_layer3(k,1) - y_k;
    end
   
    % step 3:
    % calculate delta for hidden layer 2
    Theta2_new = Theta2(:,2:end);
    d2 = Theta2_new'*d3.*sigmoidGradient(z_layer2);
    
    sum = d2*a_layer1;
    Delta1 = Delta1 + sum;
    
    sum = d3*(a_layer2');
    Delta2 = Delta2 + sum;    
end

Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;


% Part 3: Implement regularization

% set the first column of Theta1 and Theta2 to zero
Theta1(:,1) = 0;
Theta2(:,1) = 0;

% scale each Theta matrix by lambda/m
Theta1 = Theta1*(lambda/m);
Theta2 = Theta2*(lambda/m);

% add each of these modified and scaled Theta matrices to the
% un-regularized Theta gradients 

Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
