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

% compute cost J
% ======================
total_sum = 0;
for i = 1:m
    % calculate (theta transpose.x)
    X_rowi = X(i,:);
    X_rowi = X_rowi';
    theta_t_x = theta'*X_rowi;
    
    %theta_t_x = theta(1,1)*X(i,1) + theta(2,1)*X(i,2) + theta(3,1)*X(i,3);
    
    % calculate hypothesis
    h_theta = 1/(1+exp(-theta_t_x));

    % calculate partial sums for each row of training data
    sum = -y(i)*log(h_theta) - ((1-y(i))*log(1-h_theta));
    
    % accumulate partial sums    
    total_sum = total_sum + sum;
end

sum_regularized_theta = 0;

for j = 1:length(theta);
    if j == 1
        continue;
    end
    sum_regularized_theta = sum_regularized_theta + theta(j,1) * theta(j,1);
end

J = total_sum/m + (lambda/(2*m))*sum_regularized_theta;


% compute gradient
% =================
for g = 1:length(grad)
    total_sum = 0;
    for i = 1:m
        % calculate (theta transpose.x)
        X_rowi = X(i,:);
        X_rowi = X_rowi';
        theta_t_x = theta'*X_rowi;

        %theta_t_x = theta(1,1)*X(i,1) + theta(2,1)*X(i,2) + theta(3,1)*X(i,3);

        % calculate hypothesis
        h_theta = 1/(1+exp(-theta_t_x));

        % partial sum
        sum = (h_theta - y(i))*X(i,g);
        
        total_sum = total_sum + sum;
    end
      
    if g == 1
        grad(g,1) = total_sum/m;
    else
        grad(g,1) = total_sum/m + (lambda/m)*theta(g,1);
    end
end

% =============================================================

end
