function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

%calculate exp
exp_itemwise = exp(-z);

%add 1
exp_itemwise_add1 = 1.+exp_itemwise;

%divide by 1
exp_itemwise_div1 = 1./exp_itemwise_add1;

g = exp_itemwise_div1;

% =============================================================

end
