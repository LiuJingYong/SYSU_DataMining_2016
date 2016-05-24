function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%   GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
tempTheta = zeros(size(X, 2), 1);

for iter = 1:num_iters

    for i = 1:size(X, 2)
        tempTheta(i, 1) = theta(i, 1) - alpha / m * ((X * theta - y)' * X(:,i));
    end

    theta = tempTheta;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);
    if iter > 1 && J_history(iter) > J_history(iter - 1)
        alpha = alpha / 2;
        fprintf('now alpha is %f \n', alpha);
    end
    fprintf('this is %d turn...\n', iter);
end

end
