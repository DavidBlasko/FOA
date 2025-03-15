clc; clear; close all;

% Objective function - Rosenbrock's function
rosenbrock = @(x) (1 - x(1))^2 + 5 * (x(2) - x(1)^2)^2;

% Gradient - Rosenbrock's function
grad_rosenbrock = @(x) [-2*(1-x(1)) - 20*x(1)*(x(2) - x(1)^2); 10 * (x(2) - x(1)^2)];

% Starting point
x = [-1; -1];

% Params
tol = 1e-2; % Termination criteria ||grad f(x)|| < 1e-2
max_iter = 100;
max_f_calls = 1000;
Q = eye(2); % Initial approximation of Hessian inverse
trajectory_track = x'; % Store trajectory

% Init
iter = 0;
f_calls = 0;

%% Quasi-Newton DFP Method
while norm(grad_rosenbrock(x)) > tol && iter < max_iter && f_calls < max_f_calls
    d = -Q * grad_rosenbrock(x);  % Search direction
    alpha = fminbnd(@(a) rosenbrock(x + a*d), 0, 1); % Line search
    f_calls = f_calls + 1;
    
    % Update x
    s = alpha * d;
    x_new = x + s;
    y = grad_rosenbrock(x_new) - grad_rosenbrock(x);
    
    % DFP Update
    Q = Q + (s * s') / (s' * y) - (Q * y * y' * Q) / (y' * Q * y);
    
    x = x_new;
    trajectory_track = [trajectory_track; x'];
    iter = iter + 1;
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
% % Starting point
plot(trajectory_track(1,1), trajectory_track(1,2),'ko','MarkerSize',5,'MarkerFaceColor','k');
text(trajectory_track(1,1), trajectory_track(1,2) + 0.1, sprintf('Start'));
% % Solution - Optimal point
plot(trajectory_track(end,1), trajectory_track(end,2),'ko','MarkerSize',5,'MarkerFaceColor','k');
text(trajectory_track(end,1), trajectory_track(end,2) + 0.1, sprintf('Optimum'));
xlabel('x1'); ylabel('x2');
title('Quasi-Newton DFP Method');
grid on;

%% Printing
% Console output for number of iterations and func calls
disp(['Number of interations done: ', num2str(iter)]);
disp(['Number of function calls: ', num2str(f_calls)]);