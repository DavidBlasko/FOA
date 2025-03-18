clc; clear; close all;

% Objective function - Flower function
flower_func = @(x) 1 * norm(x) + 1 * sin(4 * atan2(x(2), x(1)));

% Flower function with count penalty to constraint (x1^2 + x2^2 >= 2)
flower_penalty = @(x, rho) flower_func(x) + rho * max(0, 2 - sum(x.^2));

% Params
x0 = [-2; -2];          % Starting point
P = eye(2) * 2;         % Initial covariance matrix
m = 40;                 % Population size
m_elite = 10;           % Elite samples
max_iter = 100;         % Max iterations
max_f_calls = 1000;     % Max function calls
rho = 10;               % Penalty parameter

% Init
samples = mvnrnd(x0', P, m);
objective_func_values = arrayfun(@(i) flower_penalty(samples(i, :)', rho), 1:m);
trajectory = x0'; % Store trajectory
iter_count = 0;
f_calls = m;

%% Cross-Entropy Method
while iter_count < max_iter && f_calls < max_f_calls
    
    % Sort samples by function value
    [~, order] = sort(objective_func_values);
    elite_samples = samples(order(1:m_elite), :);

    % Update mean and covariance using elite samples
    mu = mean(elite_samples);
    P = cov(elite_samples);

    % Generate new population
    samples = mvnrnd(mu, P, m);
    objective_func_values = arrayfun(@(i) flower_penalty(samples(i, :)', rho), 1:m);
    
    % Update function calls count
    f_calls = f_calls + m;

    % Store best solution of this iteration
    trajectory = [trajectory; mu];

    iter_count = iter_count + 1;
end

%% Plotting
[X1, X2] = meshgrid(linspace(-3, 3, 100), linspace(-3, 3, 100));
F_X = arrayfun(@(x1, x2) flower_func([x1; x2]), X1, X2);
hold on; grid on; axis equal;
contour(X1, X2, F_X, 'DisplayName', 'Flower Function');
xlabel('x_1'); ylabel('x_2'); title('Cross-Entropy Method w/ Penalisation & Constraints');

% Trajectory
plot(trajectory(:,1), trajectory(:,2),'-x', 'Color', '#750a72', 'DisplayName', ['ρ = ' num2str(rho)]);
% Starting point
scatter(trajectory(1,1), trajectory(1,2), 25, 'bo', 'filled', 'DisplayName', 'Start');
text(trajectory(1,1), trajectory(1,2) + 0.1, 'Start');
% Optimum point
scatter(trajectory(end,1), trajectory(end,2), 25, 'ko', 'filled', 'DisplayName', 'Optimum');
text(trajectory(end,1), trajectory(end,2) + 0.1, 'Optimum');
% Constraint boundary (x1^2 + x2^2 = 2)
th = linspace(0, 2*pi, 100);
circle_x = sqrt(2) * cos(th);
circle_y = sqrt(2) * sin(th);
plot(circle_x, circle_y, '-', 'Color', '#8d929c', 'LineWidth', 1.5, 'DisplayName', 'Constraint Boundary');
legend;

%% Printing
fprintf('Optimization completed in %d iterations and %d function calls.\n', iter_count, f_calls);
fprintf('Optimum: x = [%f, %f], f(x) = %f\n', trajectory(end,1), trajectory(end,2), flower_func(trajectory(end, :)'));
