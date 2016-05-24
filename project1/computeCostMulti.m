function J = computeCostMulti(X, y, theta)

m = length(y); % number of training examples
J = 0;

% Compute the cost of a particular choice of theta
% Set J to the cost.
J = (1 / (2 * m)) * (X * theta - y)' * (X * theta - y);

end
