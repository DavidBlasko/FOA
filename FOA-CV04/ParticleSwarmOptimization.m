clc; clear; close all;

% Objective function - Ackley's function
ackley = @(x) -20*exp(-0.2*sqrt(0.5*(x(1)^2 + x(2)^2))) - exp(0.5*(cos(2*pi*x(1)) + cos(2*pi*x(2)))) + 20 + exp(1);

% Params
pop_size = 10;    
max_iter = 100;  
max_f_calls = 1000;
w = 0.9;                        % Inertia weight
c1 = 1.2;                       % Cognitive coefficient
c2 = 1.2;                       % Social coefficient
bounds = [-6, 6; -4.5, 4.5];    % Search space bounds

% Initialize population
pop = bounds(:,1)' + (bounds(:,2)' - bounds(:,1)') .* rand(pop_size, 2);
velocities = zeros(size(pop));
p_best = pop;
p_best_values = arrayfun(@(i) ackley(pop(i, :)), 1:pop_size);
[g_best_value, best_idx] = min(p_best_values);
g_best = p_best(best_idx, :);
func_calls = pop_size;

%% Plotting setup
[X1, X2] = meshgrid(linspace(-7, 7, 100), linspace(-7, 7, 100));
F_X = arrayfun(@(x1, x2) ackley([x1, x2]), X1, X2);
hold on; grid on;
contour(X1, X2, F_X, 20);
xlabel('x_1'); ylabel('x_2'); title('Particle Swarm Optimization (PSO)');
% Initial population
scatter(pop(:,1), pop(:,2), 25, 'ko', 'filled');
text(pop(:,1), pop(:,2) + 0.1, 'InitPop');

%% Particle Swarm Optimization Loop
iter = 0;
while iter < max_iter && func_calls < max_f_calls
    r1 = rand(pop_size, 2);
    r2 = rand(pop_size, 2);

    % Update velocity and position
    velocities = w * velocities + c1 * r1 .* (p_best - pop) + c2 * r2 .* (g_best - pop);
    pop = pop + velocities;
    
    % Apply bounds
    pop = max(min(pop, bounds(:,2)'), bounds(:,1)');

    % Evaluate function
    new_values = arrayfun(@(i) ackley(pop(i, :)), 1:pop_size);
    func_calls = func_calls + pop_size;

    % Update personal best
    improved = new_values < p_best_values;
    p_best(improved, :) = pop(improved, :);
    p_best_values(improved) = new_values(improved);

    % Update global best
    [new_g_best_value, best_idx] = min(p_best_values);
    if new_g_best_value < g_best_value
        g_best_value = new_g_best_value;
        g_best = p_best(best_idx, :);
    end

    iter = iter + 1;
    
    scatter(pop(:,1), pop(:,2), 10, 'ro', 'filled'); % Update graph - population
    pause(0.05);
end

% Optimum plot
scatter(g_best(1), g_best(2), 35, 'bo', 'filled');
text(g_best(1), g_best(2) + 0.1, 'Optimum', 'FontWeight', 'bold');

%% Printing results
fprintf('Optimization completed in %d iterations and %d function calls.\n', iter, func_calls);
fprintf('Optimum: x = [%f, %f], f(x) = %f\n', g_best(1), g_best(2), g_best_value);
