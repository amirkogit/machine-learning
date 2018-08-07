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

	% initialize theta
	theta0 = theta(1,1);
	theta1 = theta(2,1);

	%calculate h_theta	
	total_sum_theta0 = 0;

	for i = 1:m
		hypo = (theta0*X(i,1) + theta1*X(i,2));
		diff = (hypo - y(i))*X(i,1);
		total_sum_theta0 = total_sum_theta0 + diff;		
	end
	
	total_sum_theta1 = 0;
	for i = 1:m
		hypo = (theta0*X(i,1) + theta1*X(i,2));
		diff = (hypo - y(i))*X(i,2);
		total_sum_theta1 = total_sum_theta1 + diff;		
	end
	
	theta0 = theta0 - alpha*(total_sum_theta0/m);
	theta1 = theta1 - alpha*(total_sum_theta1/m);
	
	%put new values of theta0 and theta1
	theta(1,1) = theta0;
	theta(2,1) = theta1;
	
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
