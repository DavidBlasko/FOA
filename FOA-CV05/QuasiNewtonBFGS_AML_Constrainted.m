clc; clear; close all;

% Objective function - Flower function
flower_func = @(x) 1 * norm(x) + 1 * sin(4 * atan2(x(2), x(1)));

% Flower function gradient
flower_grad = @(x) [(x(1) / sqrt(x(1).^2 + x(2).^2)) + (4 * cos(4 * atan2(x(2), x(1))) * (-x(2) / (x(1)^2 + x(2)^2))); (x(2) / sqrt(x(1).^2 + x(2).^2)) + (4 * cos(4 * atan2(x(2), x(1))) * (x(1) / (x(1)^2 + x(2)^2)))];

% Equality constraint (x1^2 + x2^2 = 2)
constr = @(x) x(1)^2 + x(2)^2 - 2;
constr_grad = @(x) [2*x(1); 2*x(2)];

% Params
x0 = [-2; -2];                          % Starting point
lambda = 0;                             % Initial Lagrange multiplier
rho_values = [0.01, 0.5, 1, 4, 8, 16];  % Penalty parameters
beta = 10;                              % Update factor for penalty parameter
max_iter = 100;                         % Maximum iterations
max_f_calls = 1000;                     % Maximum function calls 
tol = 1e-6;                             % Convergence tolerance
n = length(x0);
Q = eye(n);                             % Approximation of Hessian inverse

%Init
total_iter_count = 0;
total_f_calls = 0;
trajectory = x0';                       % Store trajectory
x = x0;

%% Plotting
hold on; grid on; axis equal;
xlabel('x_1'); ylabel('x_2'); title('Quasi-Newton BFGS w/ Augmented Lagrange');
% Meshgrid and contour plot
[X1, X2] = meshgrid(linspace(-3, 3, 100), linspace(-3, 3, 100));
F_X = arrayfun(@(x1, x2) flower_func([x1; x2]), X1, X2);
contour(X1, X2, F_X, 'DisplayName', 'Flower Function');
% Colors for rho values
colors = {'#750a72', '#a40f9f', '#d313cc', '#ec2ce6', '#f05bec', '#f58af1'};
color_index = 1;

%% Quasi-Newton BFGS w/ Augmented Lagrange
for rho = rho_values
    sub_trajectory = x';
    iter_count = 0;
    function_calls = 0;
    
    while iter_count < max_iter && function_calls < max_f_calls
        % Define Augmented Lagrange function and its gradient
        L = @(x) flower_func(x) + lambda * constr(x) + (rho / 2) * constr(x)^2;
        function_calls = function_calls + 1;
        total_f_calls = total_f_calls + 1;
        L_grad = @(x) flower_grad(x) + (lambda + rho * constr(x)) * constr_grad(x);
        
        g = L_grad(x);
        if norm(g) < tol
            break;
        end
        
        if iter_count == 0
            d = -g;
        else
            gamma = g - g_old;
            delta = x - x_old;
            Q = Q - (delta * gamma' * Q + Q * gamma * delta') / (delta' * gamma) + ...
                (1 + gamma' * Q * gamma / (delta' * gamma)) * (delta * delta') / (delta' * gamma);
            d = -Q * g;
        end
        
        % Line search
        alpha = fminbnd(@(a) L(x + a * d), 0, 1);
        
        % Update variables
        x_old = x;
        g_old = g;
        x = x + alpha * d;
        sub_trajectory = [sub_trajectory; x'];
        iter_count = iter_count + 1;
        total_iter_count = total_iter_count + 1;
    end
    
    trajectory = [trajectory; sub_trajectory];
    plot(sub_trajectory(:,1), sub_trajectory(:,2), '-x', 'Color', colors{color_index}, 'DisplayName', ['ρ = ' num2str(rho)]);
    color_index = mod(color_index, length(colors)) + 1;
    
    lambda = lambda + rho * constr(x);
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
plot(circle_x, circle_y, '-', 'Color', '#8d929c', 'LineWidth', 1.5 ,'DisplayName', 'Constraint Boundary');
legend;

%% Printing
fprintf('Optimization completed in %d iterations and %d function calls.\n', total_iter_count, total_f_calls);
fprintf('Optimum: x = [%f, %f], f(x) = %f\n', trajectory(end,1), trajectory(end,2), flower_func(trajectory(end, :)'));