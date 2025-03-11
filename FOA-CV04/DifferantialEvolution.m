clc; clear; close all;

% Ackley function definition
ackley = @(x) -20*exp(-0.2*sqrt(0.5*(x(1)^2 + x(2)^2))) - exp(0.5*(cos(2*pi*x(1)) + cos(2*pi*x(2)))) + 20 + exp(1);

% Parameters
pop_size = 10;
p = 0.9;                       % Crossover probability
w = 0.8;                       % Scaling factor for mutation
bounds = [-6, 6; -4.5, 4.5];
max_iter = 100;
max_func_calls = 1000;

% Initial population
pop = bounds(:,1)' + (bounds(:,2)' - bounds(:,1)') .* rand(pop_size, 2);
func_calls = pop_size;

%% Initial plotting
[X, Y] = meshgrid(linspace(bounds(1,1), bounds(1,2), 100), linspace(bounds(2,1), bounds(2,2), 100));
Z = arrayfun(@(x, y) ackley([x, y]), X, Y);
figure; hold on; grid on;
contour(X, Y, Z, 20);
xlabel('x_1'); ylabel('x_2'); title('Differential Evolution');

% Initial population
scatter(pop(:,1), pop(:,2), 25, 'ko', 'filled');
text(pop(:,1), pop(:,2) + 0.1, 'InitPop');

%% Differential Evolution
iter = 0;
best_fitness = inf;
best_solution = [];

while iter < max_iter && func_calls < max_func_calls
    new_pop = pop;
    
    for i = 1:pop_size
        % Mutation: Select three distinct random individuals (excluding i)
        idxs = randperm(pop_size);
        idxs = idxs(idxs ~= i); % Remove current index
        a = pop(idxs(1), :);
        b = pop(idxs(2), :);
        c = pop(idxs(3), :);
        
        mutant = a + w * (b - c); % Compute mutant vector

        % Crossover
        cross_points = rand(1,2) < p;
        if ~any(cross_points)
            cross_points(randi(2)) = true;
        end
        trial = pop(i, :);
        trial(cross_points) = mutant(cross_points);

        % Selection with optimized function call
        trial_fitness = ackley(trial);  % Compute trial fitness only once
        func_calls = func_calls + 1;
        
        if trial_fitness < ackley(pop(i, :))
            new_pop(i, :) = trial;
        end
    end

    pop = new_pop;
    iter = iter + 1;

    % Find best solution in current population
    [current_best_fitness, idx] = min(arrayfun(@(row) ackley(pop(row, :)), 1:pop_size));
    if current_best_fitness < best_fitness
        best_fitness = current_best_fitness;
        best_solution = pop(idx, :);
    end

    scatter(pop(:,1), pop(:,2), 10, 'ro', 'filled'); % Update graph - population
    pause(0.1);
end

% Optimum plotting
scatter(best_solution(1), best_solution(2), 50, 'bo', 'filled');
text(best_solution(1), best_solution(2) + 0.1, 'Optimum','FontWeight','bold');

%% Printing
fprintf('Optimization completed in %d iterations and %d function calls.\n', iter, func_calls);
fprintf('Optimum: x = [%.4f, %.4f], f(x) = %.4f\n', best_solution(1), best_solution(2), best_fitness);
