function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:
%
%       for i = 1:length(lambda_vec)
%           lambda = lambda_vec(i);
%           % Compute train / val errors when training linear 
%           % regression with regularization parameter lambda
%           % You should store the result in error_train(i)
%           % and error_val(i)
%           ....
%           
%       end
%
%

for i = 1:length(lambda_vec)
    lambda = lambda_vec(i);

    X_subset = X(:,:); % include all training set examples
    y_subset = y(:); % include all training set examples

    theta = trainLinearReg(X_subset,y_subset,lambda);
    
    m_subset = size(X, 1); % total number of training examples in this iteration (includes all training examples)

    % compute training error
    J_train = ((X_subset*theta - y_subset)'*(X_subset*theta - y_subset))/(2*m_subset);

    m_subset_val = size(Xval,1);
    
    % compute validation error
    J_val = ((Xval*theta - yval)'*(Xval*theta - yval))/(2*m_subset_val);
    
    error_train(i) = J_train;
    error_val(i) = J_val;
end

% =========================================================================

end
