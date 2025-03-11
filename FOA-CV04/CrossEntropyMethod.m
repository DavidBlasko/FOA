clc; clear; close all;

% Objective function - Ackley's function
ackley = @(x) -20*exp(-0.2*sqrt(0.5*(x(1).^2 + x(2).^2))) - exp(0.5*(cos(2*pi*x(1)) + cos(2*pi*x(2)))) + 20 + exp(1);

% Params
x0 = [-6; -4.5];       % Starting point
P = eye(2) * 10;       % Initial covariance, proposal distribution
m = 40;                % Population size
m_elite = 10;          % Elite samples
max_iter = 100;        % Max iterations
max_func_calls = 1000; % Max function calls

% Init
samples = mvnrnd(x0', P, m);
func_vals = arrayfun(@(i) ackley(samples(i, :)'), 1:m);
func_calls = m;
trajectory = x0'; % Store trajectory
iter_count = 0;

%% Cross-Entropy Method
while iter_count < max_iter && func_calls < max_func_calls
    iter_count = iter_count + 1;

    % Sort samples by function value
    [~, order] = sort(func_vals);
    elite_samples = samples(order(1:m_elite), :);

    % Update mean and covariance using elite samples
    mu = mean(elite_samples);
    P = cov(elite_samples);

    % Generate new populationbest_solution
    samples = mvnrnd(mu, P, m);
    func_vals = arrayfun(@(i) ackley(samples(i, :)'), 1:m);
    
    % Update function calls count
    func_calls = func_calls + m;

    % Store best solution of this iteration
    trajectory = [trajectory; mu]; 
end

%% Plotting
[X, Y] = meshgrid(linspace(-7, 7, 100), linspace(-7, 7, 100));
Z = arrayfun(@(x, y) ackley([x; y]), X, Y);
figure; hold on; grid on;
contour(X, Y, Z, 20);
xlabel('x_1'); ylabel('x_2'); title('Cross-Entropy Method');

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
fprintf('Optimization completed in %d iterations and %d function calls.\n', iter_count, func_calls);
fprintf('Optimum: x = [%f, %f], f(x) = %f\n', trajectory(end,1), trajectory(end,2), ackley(trajectory(end, :)));
