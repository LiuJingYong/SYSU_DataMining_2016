
%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = csvread('train-1.csv');
X = data(:, 2:size(data, 2)-1);
y = data(:, size(data, 2));
m = length(y);
pre = ones(m, 1);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================

fprintf('Running gradient descent ...\n');

% Choose some alpha value 
alpha = 0.1;
num_iters = 5000;	% here change the iteration times

% Init Theta and Run Gradient Descent 
theta = zeros(size(X,2), 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');


%% ================ Part 3: Predict the Test Set ================

predict = csvread('test-1.csv');
predict(:,1) = 1;
pre = predict * theta;
A = ones(25000,1);
for kk = 1:25000
    A(kk, 1) = kk - 1;
end
anss = [A pre]
csvwrite('ans.csv', anss, 1, 0);

