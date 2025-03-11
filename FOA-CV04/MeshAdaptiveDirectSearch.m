clc; clear; close all;

% Objective function - Ackley's function
ackley = @(x) -20*exp(-0.2*sqrt(0.5*(x(1).^2 + x(2).^2))) - exp(0.5*(cos(2*pi*x(1)) + cos(2*pi*x(2)))) + exp(1) + 20;

% Parameters
x0 = [-6; -4.5];        % Starting point
epsilon = 1e-6;         % Tolerance
max_iter = 100;         % Maximum number of iterations
max_func_calls = 1000;  % Maximum function evaluations

% Initialization
x = x0;
y = ackley(x);
alpha = 1.0;  
n = length(x);  

trajectory = x';
iter_count = 0;
func_calls = 1;

%% Mesh Adaptive Direct Search (MADS)
while alpha > epsilon && iter_count < max_iter && func_calls < max_func_calls
    iter_count = iter_count + 1;
    improved = false;
    
    % Generate a random positive spanning set with step size alpha
    D = rand_positive_spanning_set(alpha, n);
    
    % Search in all directions
    for i = 1:size(D,2)
        x_candidate = x + alpha * D(:,i);
        y_candidate = ackley(x_candidate);
        func_calls = func_calls + 1;
        
        if y_candidate < y
            x = x_candidate;
            y = y_candidate;
            improved = true;
            
            % Second step: check larger step in same direction
            x_candidate = x + 3 * alpha * D(:,i);
            y_candidate = ackley(x_candidate);
            func_calls = func_calls + 1;
            
            if y_candidate < y
                x = x_candidate;
                y = y_candidate;
            end
            break;
        end
        
        % Stop if max function calls reached
        if func_calls >= max_func_calls
            break;
        end
    end
    
    % Store trajectory
    trajectory = [trajectory; x'];

    % Adapt alpha
    if improved
        alpha = min(4 * alpha, 1); % Increase step size
    else
        alpha = alpha / 4; % Reduce step size
    end
end

%% Generate Random Positive Spanning Set
function D = rand_positive_spanning_set(alpha, n)
    delta = round(1 / sqrt(alpha));  
    L = tril((2 * rand(n) - 1) * delta);  

    % Randomly permute rows and columns
    idx = randperm(n);
    L = L(idx, idx);

    % Add the negative sum of all rows as the last column to ensure a spanning set
    D = [L, -sum(L, 2)];
end

%% Plotting
[X, Y] = meshgrid(linspace(-7, 7, 100), linspace(-7, 7, 100));
Z = arrayfun(@(x, y) ackley([x; y]), X, Y);
figure; hold on; grid on;
contour(X, Y, Z, 20);
xlabel('x_1'); ylabel('x_2'); title('Mesh Adaptive Direct Search (MADS)');

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
fprintf('Optimum: x = [%f, %f], f(x) = %f\n', x(1), x(2), ackley(trajectory(end, :)));
