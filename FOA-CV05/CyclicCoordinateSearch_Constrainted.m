clc; clear; close all;

% Objective function - Flower function
flower_func = @(x) 1 * norm(x) + 1 * sin(4 * atan2(x(2), x(1)));
% Flower function with penalty to constraint
flower_penalty = @(x, rho) flower_func(x) + rho * max(0, 2 - sum(x.^2));

% Starting point
x = [-2; -2];

% Params
rho_values = [0.5, 1, 5, 10];   % Penalty parameters
max_iter = 100;                 % Max iterations
max_f_calls = 1000;             % Max function calls
tol = 1e-6;                     % Tolerance

% Init
n = length(x);
trajectory = x'; % Store trajectory
iter = 0;
f_calls = 0;

%% Cyclic Coordinate Search w/ Penalisation
for rho = rho_values
    while iter < max_iter && f_calls < max_f_calls
        x_old = x;
        for i = 1:n
            d = zeros(n, 1); % Direction
            d(i) = 1; % Base vector
            alpha = fminbnd(@(alpha) flower_penalty(x + alpha * d, rho), 0, 1);
            x = x + alpha * d;
            f_calls = f_calls + 1;
            trajectory = [trajectory; x'];
        end
        
        % Accelerated step between new and old points
        d_accel = x - x_old;
        if norm(d_accel) > 0
            alpha = fminbnd(@(alpha) flower_penalty(x + alpha * d_accel, rho), 0, 1);
            x = x + alpha * d_accel;
            f_calls = f_calls + 1;
            trajectory = [trajectory; x'];
        end
        
        % Ending condition to meet tolerance
        if norm(x - x_old) < tol
            break;
        end

        iter = iter + 1;
    end
end

%% Plotting
[X1, X2] = meshgrid(-3:0.05:3, -3:0.05:3);
Z = arrayfun(@(x1, x2) flower_func([x1; x2]), X1, X2);
hold on; grid on;
contour(X1, X2, Z, 50);
xlabel('x_1'); ylabel('x_2'); title('Cyclic Coordinate Search w/ Penalisation & Constraints');

% Trajectory points
scatter(trajectory(:,1), trajectory(:,2), 50, 'rx');
% Trajectory line
plot(trajectory(:,1), trajectory(:,2),'b-');
% Starting point
scatter(trajectory(1,1), trajectory(1,2), 25, 'ko', 'filled');
text(trajectory(1,1), trajectory(1,2) + 0.1, 'Start');
% Solution - Optimal point
scatter(trajectory(end,1), trajectory(end,2), 25, 'ko', 'filled');
text(trajectory(end,1), trajectory(end,2) + 0.1, 'Optimum');
% Constraint boundary (x1^2 + x2^2 = 2)
th = linspace(0, 2*pi, 100);
circle_x = sqrt(2) * cos(th);
circle_y = sqrt(2) * sin(th);
boundary = plot(circle_x, circle_y, 'm-');
legend(boundary,{'Constraint Boundary'});

%% Printing
fprintf('Optimization completed in %d iterations and %d function calls.\n', iter, f_calls);
fprintf('Optimum: x = [%f, %f], f(x) = %f\n', trajectory(end,1), trajectory(end,2), flower_func(trajectory(end, :)));