clc; clear; close all;

% Init
x0 = [-1; -1];
max_iter = 100;
tol = 1e-2;

%% Gradient Descent w/ Decaying Stepsize
function [traj, iter_count] = gradient_descent_decaying_stepsize(x0, max_iter, tol)
    x = x0;
    traj = x;
    iter_count = 0;

    for k = 1:max_iter
        grad = gradient(x);
        if norm(grad) < tol
            break;
        end
        alpha = 0.9^k;
        x = x - alpha * (grad / norm(grad));
        traj = [traj, x];
        iter_count = iter_count + 1;
    end
end

% Objective function - Rosenbrock's function
function f = rosenbrock(x)
    a = 1;
    b = 5;
    f = (a - x(1))^2 + b * (x(2) - x(1)^2)^2;
end

% Gradient for Rosenbrock's function
function g = gradient(x)
    a = 1;
    b = 5;
    g = [-2 * (a - x(1)) - 4 * b * x(1) * (x(2) - x(1)^2);
         2 * b * (x(2) - x(1)^2)];
end

%% Plotting
[X1, X2] = meshgrid(linspace(-2, 2, 100), linspace(-1, 3, 100));
F_X = arrayfun(@(x1, x2) rosenbrock([x1; x2]), X1, X2);
contour(X1, X2, F_X, logspace(-1, 3, 20), 'DisplayName', 'Rosenbrocks Function');
hold on; grid on; axis equal;
xlabel('x_1'); ylabel('x_2'); title('Gradient Descent w/ Decaying Stepsize');

[traj_grad_desc_decaying, iter_count] = gradient_descent_decaying_stepsize(x0, max_iter, tol);
plot(traj_grad_desc_decaying(1, :), traj_grad_desc_decaying(2, :), 'rx-', 'DisplayName', 'Trajectory');

legend;

%% Printing
fprintf('Optimization completed in %d iterations.\n', iter_count);
fprintf('Optimum: x = [%f, %f], f(x) = %f\n', traj_grad_desc_decaying(1, end), traj_grad_desc_decaying(2, end), rosenbrock(traj_grad_desc_decaying(:, end)));
