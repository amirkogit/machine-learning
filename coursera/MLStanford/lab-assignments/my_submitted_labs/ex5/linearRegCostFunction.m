function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% part 1: calculate cost function (vectorized)
% ----------------------------------------------

% unregularized cost function
J = ((X*theta - y)'*(X*theta - y))/(2*m)

% regularized parameter
sum_theta = 0;
for i = 2:size(theta)
    sum_theta = sum_theta + theta(i)*theta(i);
end
reg = (lambda/(2*m))*(sum_theta);

% add regualarized parameter
 J = J + reg;

% part2 : calculate gradient (vectorized)
% -----------------------------------------
X_theta = X*theta;

grad = (X'*(X_theta - y))/m;

temp = theta; 

temp(1) = 0;   % because we don't add anything for j = 0  

grad_reg = (lambda/m)*temp;

grad = grad + grad_reg;

% =========================================================================

grad = grad(:);

end
