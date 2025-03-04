clc; clear; close all;

% Objective function - Rosenbrock's function
rosenbrock_func = @(x) (1 - x(1))^2 + 5 * (x(2) - x(1)^2)^2;

% Init of simplex params
initial_simplex = [-1, -1; -0.8, -0.8; -0.5, -1];
n = size(initial_simplex, 2); % Number of dimensions (2)
max_iter = 100;
max_f_calls = 1000;
delta = 1e-6; % Tolerance for ending condition

simplex = initial_simplex;
f_values = arrayfun(@(i) rosenbrock_func(simplex(i,:)), 1:size(simplex, 1));
trajectory_track = simplex;
f_calls = numel(f_values);
iter = 0;

%% Nelder-Mead Simplex Method (reflection, expansion, contraction, shrinkage)
while iter < max_iter && f_calls < max_f_calls
    [f_values, idx] = sort(f_values);
    simplex = simplex(idx, :);
    
    % Convergency check
    if std(f_values) < delta
        break;
    end
    
    % Center of gravity
    centroid = mean(simplex(1:end-1, :), 1);
    worst = simplex(end, :);
    
    % Reflection
    xr = centroid + (centroid - worst);
    fr = rosenbrock_func(xr);
    f_calls = f_calls + 1;
    
    if fr < f_values(1)
        % Expansion
        xe = centroid + 2 * (xr - centroid);
        fe = rosenbrock_func(xe);
        f_calls = f_calls + 1;
        
        if fe < fr
            simplex(end, :) = xe;
            f_values(end) = fe;
        else
            simplex(end, :) = xr;
            f_values(end) = fr;
        end
        
    elseif fr < f_values(2)
        simplex(end, :) = xr;
        f_values(end) = fr;
    else
        % Contraction
        xc = centroid + 0.5 * (worst - centroid);
        fc = rosenbrock_func(xc);
        f_calls = f_calls + 1;
        
        if fc < f_values(end)
            simplex(end, :) = xc;
            f_values(end) = fc;
        else
            % Shrinkage of simplex
            simplex(2:end, :) = simplex(1, :) + 0.5 * (simplex(2:end, :) - simplex(1, :));
            f_values(2:end) = arrayfun(@(i) rosenbrock_func(simplex(i,:)), 2:size(simplex, 1));
        end
    end
    
    trajectory_track = cat(1, trajectory_track, simplex);
    iter = iter + 1;
end

optimum = simplex(1, :);

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
plot(optimum(1), optimum(2),'o','MarkerSize',5,'MarkerFaceColor','k');
text(optimum(1), optimum(2) + 0.1,'Optimum');
xlabel('x1'); ylabel('x2');
title('Nelder-Mead Simplex Method');
grid on;

%% Printing
% Console output for number of iterations and func calls
disp(['Number of interations done: ', num2str(iter)]);
disp(['Number of function calls: ', num2str(f_calls)]);