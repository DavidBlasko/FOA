clc; clear; close all;

% Objective function - Rosenbrock's function
rosenbrock_func = @(x) (1 - x(1))^2 + 5 * (x(2) - x(1)^2)^2;

% Starting point
x = [-1, -1];

% Parameters
alpha = 1; % Step size
epsilon = 1e-6; % Tolerance
gamma = 0.5; % Step reduction factor
max_iter = 100; % Max number of iterations
max_f_calls = 1000; % Max number of function evaluations
sgn = [-1, 1]; % Search directions
basis_vec = [1, 0; 0, 1]; % Basis vectors for x and y directions

% Initialization
n = length(x);
y = rosenbrock_func(x);
iter = 0;
f_calls = 0;
trajectory_track = x; % Store trajectory points
hold on;

%% Hooke-Jeeves Method
while alpha > epsilon && iter < max_iter && f_calls < max_f_calls
    x_best = x;
    y_best = y;
    improvement = false;

    % Exploratory search
    for i = 1:n
        for l = sgn
            x_new = x + l * alpha * basis_vec(i, :);
            y_new = rosenbrock_func(x_new);
            f_calls = f_calls + 1;
            plot(x_new(1),x_new(2),'cx'); % Mark directions

            if y_new < y_best
                x_best = x_new; 
                y_best = y_new;
                improvement = true;
            end
        end
    end

    if improvement
        x = x_best;
        y = y_best;
        trajectory_track = [trajectory_track; x]; % Store trajectory
    else
        alpha = alpha * gamma; % Reduce step size
    end

    iter = iter + 1;
end

%% Plotting
[X1, X2] = meshgrid(-2:0.05:2, -3:0.05:3);
F = (1 - X1).^2 + 5 * (X2 - X1.^2).^2;
contour(X1, X2, F, 50);

% Trajectory points
plot(trajectory_track(:,1), trajectory_track(:,2),'rx');
% Trajectory line
plot(trajectory_track(:, 1), trajectory_track(:, 2),'b--');
% Starting point
plot(trajectory_track(1,1), trajectory_track(1,2),'ko','MarkerSize',5,'MarkerFaceColor','k');
text(trajectory_track(1,1), trajectory_track(1,2) + 0.1, sprintf('Start'));
% Solution - Optimal point
plot(trajectory_track(end,1), trajectory_track(end,2),'ko','MarkerSize',5,'MarkerFaceColor','k');
text(trajectory_track(end,1), trajectory_track(end,2) + 0.1, sprintf('Optimum'));
xlabel('x1'); ylabel('x2'); title('Hooke-Jeeves Method');
grid on;
hold off;

%% Printing
% Console output for number of iterations and func calls
disp(['Number of interations done: ', num2str(iter)]);
disp(['Number of function calls: ', num2str(f_calls)]);