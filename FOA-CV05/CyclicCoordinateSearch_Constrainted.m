clc; clear; close all;

% Objective function - Flower function
flower_func = @(x) 1 * norm(x) + 1 * sin(4 * atan2(x(2), x(1)));

% Flower function with count penalty to constraint (x1^2 + x2^2 >= 2)
flower_penalty = @(x, rho) flower_func(x) + rho * max(0, 2 - sum(x.^2));

% Params
x0 = [-2; -2];                  % Starting point
rho_values = [0.5, 1, 5, 10];   % Penalty parameters
max_iter = 100;                 % Max iterations
max_f_calls = 1000;             % Max function calls
tol = 1e-6;                     % Tolerance

% Init
n = length(x0);
trajectory = x0';               % Store trajectory
x = x0;                         % Starting point of trajectory
total_iter_count = 0;            % Iters through all rhos
total_f_calls = 0;               % Func calls through all rhos

%% Plotting
hold on; grid on; axis equal;
xlabel('x_1'); ylabel('x_2'); title('Cyclic Coordinate Search w/ Penalisation & Constraints');
% Meshgrid and contour plot
[X1, X2] = meshgrid(linspace(-3, 3, 100), linspace(-3, 3, 100));
F_X = arrayfun(@(x1, x2) flower_func([x1; x2]), X1, X2);
contour(X1, X2, F_X, 'DisplayName', 'Flower Function');
% Colors for rho values
colors = {'#d313cc', '#ec2ce6', '#f05bec', '#f58af1'};
color_index = 1;

%% Cyclic Coordinate Search w/ Penalisation
for rho = rho_values
    iter_count = 0;
    f_calls = 0;
    sub_trajectory = x'; % Store sub-trajectory for each rho
    while iter_count < max_iter && f_calls < max_f_calls
        x_old = x;
        for i = 1:n
            d = zeros(n, 1); % Direction
            d(i) = 1; % Base vector
            alpha = fminbnd(@(alpha) flower_penalty(x + alpha * d, rho), 0, 1);
            x = x + alpha * d;
            f_calls = f_calls + 1;
            total_f_calls = total_f_calls + 1;
            sub_trajectory = [sub_trajectory; x'];
        end
        
        % Accelerated step between new and old points
        d_accel = x - x_old;
        if norm(d_accel) > 0
            alpha = fminbnd(@(alpha) flower_penalty(x + alpha * d_accel, rho), 0, 1);
            x = x + alpha * d_accel;
            f_calls = f_calls + 1;
            total_f_calls = total_f_calls + 1;
            sub_trajectory = [sub_trajectory; x'];
        end
        
        % Ending condition to meet tolerance
        if norm(x - x_old) < tol
            break;
        end

        iter_count = iter_count + 1;
        total_iter_count = total_iter_count + 1;
    end
    trajectory = [trajectory; sub_trajectory];
    plot(sub_trajectory(:,1), sub_trajectory(:,2),'-x', 'Color', colors{color_index}, 'DisplayName', ['ρ = ' num2str(rho)]);
    color_index = mod(color_index, length(colors)) + 1;
end

%% Plotting
% Starting point
scatter(trajectory(1,1), trajectory(1,2), 25, 'bo', 'filled', 'DisplayName', 'Start');
text(trajectory(1,1), trajectory(1,2) + 0.1, 'Start');
% Solution - Optimal point
scatter(trajectory(end,1), trajectory(end,2), 25, 'ko', 'filled', 'DisplayName', 'Optimum');
text(trajectory(end,1), trajectory(end,2) + 0.1, 'Optimum');
% Constraint boundary (x1^2 + x2^2 = 2)
th = linspace(0, 2*pi, 100);
circle_x = sqrt(2) * cos(th);
circle_y = sqrt(2) * sin(th);
plot(circle_x, circle_y, '-', 'Color', '#8d929c', 'LineWidth', 1.5, 'DisplayName', 'Constraint Boundary');
legend;

%% Printing
fprintf('Optimization completed in %d iterations and %d function calls.\n', total_iter_count, total_f_calls);
fprintf('Optimum: x = [%f, %f], f(x) = %f\n', trajectory(end,1), trajectory(end,2), flower_func(trajectory(end, :)'));