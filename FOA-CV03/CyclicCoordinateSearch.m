clc; clear; close all;

% Objective function - Rosenbrock's function
rosenbrock_func = @(x) (1 - x(1))^2 + 5 * (x(2) - x(1)^2)^2;

% Starting point
x = [-1; -1];

% Params
max_iter = 100; % Max number of iterations
max_f_calls = 1000; % Max number of function calls
delta = 1e-4; % Tolerance

% Init
n = length(x);
trajectory_track = x';
iter = 0;
f_calls = 0;

%% Cyclic Coordinate Search w/ Acceleration
while iter < max_iter && f_calls < max_f_calls
    x_old = x;
    for i = 1:n
        d = zeros(n, 1); % Direction
        d(i) = 1; % Base vector
        alpha = line_search(rosenbrock_func, x, d);
        x = x + alpha * d;
        f_calls = f_calls + 1;
        trajectory_track = [trajectory_track; x'];
    end
    
    % Accelerated step between new and old ponits
    d_accel = x - x_old;
    if norm(d_accel) > 0
        alpha = line_search(rosenbrock_func, x, d_accel);
        x = x + alpha * d_accel;
        f_calls = f_calls + 1;
        trajectory_track = [trajectory_track; x'];
    end
    
    % Ending condition to meet delta tolerance
    if norm(x - x_old) < delta
        break;
    end

    iter = iter + 1;

end

% Line Search function for optimal step
function [alpha] = line_search(f, x, p)
    alpha = 1;
    c = 1e-4;
    rho = 0.9;
    while f(x + alpha * p) > f(x) + c * alpha * (p' * p)
        alpha = rho * alpha;
    end
end

%% Plotting
[X1, X2] = meshgrid(-2:0.05:2, -3:0.05:3);
F = (1 - X1).^2 + 5 * (X2 - X1.^2).^2;
contour(X1, X2, F, 50);

hold on;
% Trajectory points
plot(trajectory_track(:,1), trajectory_track(:,2),'rx');
% Trajectory line
plot(trajectory_track(:,1), trajectory_track(:,2),'b--');
% Starting point
plot(trajectory_track(1,1), trajectory_track(1,2),'ko','MarkerSize',5,'MarkerFaceColor','k');
text(trajectory_track(1,1), trajectory_track(1,2) + 0.1, sprintf('Start'));
% Solution - Optimal point
plot(trajectory_track(end,1), trajectory_track(end,2),'ko','MarkerSize',5,'MarkerFaceColor','k');
text(trajectory_track(end,1), trajectory_track(end,2) + 0.1, sprintf('Optimum'));
xlabel('x1');
ylabel('x2');
title('Accelerated Cyclic Coordinate Search');
grid on;
hold off;

%% Printing
% Console output for number of iterations and func calls
disp(['Number of interations done: ', num2str(iter)]);
disp(['Number of function calls: ', num2str(f_calls)]);