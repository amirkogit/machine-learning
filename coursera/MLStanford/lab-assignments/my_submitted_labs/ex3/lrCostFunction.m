function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
% compute cost J (vectorized implementation)
% =============================================
total_sum = 0;

% compute X*theta

X_theta = sigmoid(X*theta);

total_sum = -((log(X_theta))'*y + log((1-X_theta))'*(1-y))/m;

sum_regularized_theta = 0;

for j = 1:length(theta);
    if j == 1
        continue;
    end
    sum_regularized_theta = sum_regularized_theta + theta(j,1) * theta(j,1);
end

J = total_sum + (lambda/(2*m))*sum_regularized_theta;


% ============================================================
% compute gradient (vectorized implementation)

grad = (X'*(X_theta - y))/m;

temp = theta; 

temp(1) = 0;   % because we don't add anything for j = 0  

grad_reg = (lambda/m)*temp;

grad = grad + grad_reg;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% total_sum = 0;
% for i = 1:m
%     % calculate (theta transpose.x)
%     X_rowi = X(i,:);
%     X_rowi = X_rowi';
%     theta_t_x = theta'*X_rowi;
%        
%     % calculate hypothesis
%     h_theta = 1/(1+exp(-theta_t_x));
% 
%     % calculate partial sums for each row of training data
%     sum = -y(i)*log(h_theta) - ((1-y(i))*log(1-h_theta));
%     
%     % accumulate partial sums    
%     total_sum = total_sum + sum;
% end
% 
% sum_regularized_theta = 0;

% for j = 1:length(theta);
%     if j == 1
%         continue;
%     end
%     sum_regularized_theta = sum_regularized_theta + theta(j,1) * theta(j,1);
% end
% 
% J = total_sum/m + (lambda/(2*m))*sum_regularized_theta;


% compute gradient
% =================
% for g = 1:length(grad)
%     total_sum = 0;
%     for i = 1:m
%         % calculate (theta transpose.x)
%         X_rowi = X(i,:);
%         X_rowi = X_rowi';
%         theta_t_x = theta'*X_rowi;
% 
%         % calculate hypothesis
%         h_theta = 1/(1+exp(-theta_t_x));
% 
%         % partial sum
%         sum = (h_theta - y(i))*X(i,g);
%         
%         total_sum = total_sum + sum;
%     end
%       
%     if g == 1
%         grad(g,1) = total_sum/m;
%     else
%         grad(g,1) = total_sum/m + (lambda/m)*theta(g,1);
%     end
% end

% =============================================================

grad = grad(:);

end
