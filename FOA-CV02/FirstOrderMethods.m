function FirstOrderMethods
    % init
    x0 = [-1; -1];
    max_iter = 100;
    tol = 1e-2;
    % optimalizacne metody
    traj_grad_desc_optimal = gradient_descent_optimal_stepsize(x0, max_iter, tol);
    traj_grad_desc_decaying = gradient_descent_decaying_stepsize(x0, max_iter, tol);
    traj_conjug_grad = conjugate_gradient(x0, max_iter, tol);
    % visu
    plot_trajectories(traj_grad_desc_optimal, traj_grad_desc_decaying, traj_conjug_grad);
end

function f = rosenbrock(x)
    a = 1;
    b = 5;
    f = (a - x(1))^2 + b * (x(2) - x(1)^2)^2;
end

function g = gradient(x)
    a = 1;
    b = 5;
    g = [-2 * (a - x(1)) - 4 * b * x(1) * (x(2) - x(1)^2);
         2 * b * (x(2) - x(1)^2)];
end

function alpha = golden_section_search(f, x, d, a, b, tol)
    phi = (1 + sqrt(5)) / 2;
    resphi = 2 - phi;

    x1 = a + resphi * (b - a);
    x2 = b - resphi * (b - a);
    f1 = f(x + x1 * d);
    f2 = f(x + x2 * d);

    while abs(b - a) > tol
        if f1 < f2
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + resphi * (b - a);
            f1 = f(x + x1 * d);
        else
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = b - resphi * (b - a);
            f2 = f(x + x2 * d);
        end
    end

    alpha = (a + b) / 2;
end

function traj = gradient_descent_optimal_stepsize(x0, max_iter, tol)
    x = x0;
    traj = x;

    for k = 1:max_iter
        grad = gradient(x);
        if norm(grad) < tol
            break;
        end
        alpha = golden_section_search(@rosenbrock, x, -grad, 0, 1, 1e-5);
        x = x - alpha * grad;
        traj = [traj, x];
    end
end

function traj = gradient_descent_decaying_stepsize(x0, max_iter, tol)
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

function traj = conjugate_gradient(x0, max_iter, tol)
    x = x0;
    traj = x;
    grad = gradient(x);
    d = -grad;
    iter_count = 0;

    for k = 1:max_iter
        if norm(grad) < tol
            break;
        end
        alpha = golden_section_search(@rosenbrock, x, d, 0, 1, 1e-5);
        x = x + alpha * d;
        new_grad = gradient(x);
        beta = (new_grad' * (new_grad - grad)) / (grad' * grad);
        d = -new_grad + beta * d;
        grad = new_grad;
        traj = [traj, x];
        iter_count = iter_count + 1;
    end
end

function plot_trajectories(traj_grad_desc_optimal, traj_grad_desc_decaying, traj_conjug_grad)
    [X, Y] = meshgrid(linspace(-2, 2, 100), linspace(-1, 3, 100));
    Z = arrayfun(@(x, y) rosenbrock([x; y]), X, Y);

    figure;
    contour(X, Y, Z, logspace(-1, 3, 20), 'k'); hold on;
    plot(traj_grad_desc_optimal(1, :), traj_grad_desc_optimal(2, :), 'rx-', 'DisplayName', 'Gradient Descent - Optimal Step');
    plot(traj_grad_desc_decaying(1, :), traj_grad_desc_decaying(2, :), 'gx-', 'DisplayName', 'Gradient Descent - Decaying Step');
    plot(traj_conjug_grad(1, :), traj_conjug_grad(2, :), 'bx-', 'DisplayName', 'Conjugate Gradient');
    legend;
    xlabel('x_1');
    ylabel('x_2');
    title('First Order Methods - Trajectories');
    hold off;
end