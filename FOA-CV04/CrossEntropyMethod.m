clc; clear; close all;

% Objective function - Ackley's function
ackley = @(x) -20*exp(-0.2*sqrt(0.5*(x(1).^2 + x(2).^2))) - exp(0.5*(cos(2*pi*x(1)) + cos(2*pi*x(2)))) + 20 + exp(1);

% Params
x0 = [-6; -4.5];       % Starting point
P = eye(2) * 10;       % Initial covariance, proposal distribution
m = 40;                % Population size
m_elite = 10;          % Elite samples
max_iter = 100;        % Max iterations
max_f_calls = 1000; % Max function calls

% Init
samples = mvnrnd(x0', P, m);
objective_func_values = arrayfun(@(i) ackley(samples(i, :)'), 1:m);
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

    % Generate new populationbest_solution
    samples = mvnrnd(mu, P, m);
    objective_func_values = arrayfun(@(i) ackley(samples(i, :)'), 1:m);
    
    % Update function calls count
    f_calls = f_calls + m;

    % Store best solution of this iteration
    trajectory = [trajectory; mu];

    iter_count = iter_count + 1;
end

%% Plotting
[X1, X2] = meshgrid(linspace(-7, 7, 100), linspace(-7, 7, 100));
F_X = arrayfun(@(x1, x2) ackley([x1; x2]), X1, X2);
hold on; grid on;
contour(X1, X2, F_X, 20);
xlabel('x_1'); ylabel('x_2'); title('Cross-Entropy Method (CEM)');

% Trajectory points
scatter(trajectory(:,1), trajectory(:,2), 25, 'ro', 'filled');
% Trajectory line
plot(trajectory(:,1), trajectory(:,2),'b-');
% Starting point
scatter(trajectory(1,1), trajectory(1,2), 25, 'ko', 'filled');
text(trajectory(1,1), trajectory(1,2) + 0.1, 'Start');
% Optimum point
scatter(trajectory(end,1), trajectory(end,2), 25, 'ko', 'filled');
text(trajectory(end,1), trajectory(end,2) + 0.1, 'Optimum');

%% Printing
fprintf('Optimization completed in %d iterations and %d function calls.\n', iter_count, f_calls);
fprintf('Optimum: x = [%f, %f], f(x) = %f\n', trajectory(end,1), trajectory(end,2), ackley(trajectory(end, :)));
