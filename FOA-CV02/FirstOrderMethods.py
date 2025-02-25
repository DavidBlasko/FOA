import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

def rosenbrock(x, a=1, b=5):
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

def gradient(x, a=1, b=5):
    dfdx1 = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0]**2)
    dfdx2 = 2 * b * (x[1] - x[0]**2)
    return np.array([dfdx1, dfdx2])

def optimal_step(f, x, d):
    obj = lambda alpha: f(x + alpha * d)
    res = minimize_scalar(obj, bounds=(0, 1), method='bounded')
    return res.x

def gradient_descent_optimal_stepsize(x0, max_iter=100, tol=1e-2):
    x = np.array(x0)
    trajectory = [x.copy()]
    
    for k in range(max_iter):
        grad = gradient(x)
        if np.linalg.norm(grad) < tol:
            break
        alpha = optimal_step(rosenbrock, x, -grad)  # optimalny step
        x = x - (alpha * grad)
        trajectory.append(x.copy())
    
    return np.array(trajectory)

def gradient_descent_decaying_stepsize(x0, max_iter=100, tol=1e-2):
    x = np.array(x0)
    trajectory = [x.copy()]
    
    for k in range(max_iter):
        grad = gradient(x)
        if np.linalg.norm(grad) < tol:
            break
        alpha = 0.9 ** k  # decaying step
        x = x - (alpha * (grad / np.linalg.norm(grad)))
        trajectory.append(x.copy())
    
    return np.array(trajectory)

def conjugate_gradient(x0, max_iter=100, tol=1e-2):
    x = np.array(x0)
    trajectory = [x.copy()]
    grad = gradient(x)
    d = -grad
    
    for k in range(max_iter):
        if np.linalg.norm(grad) < tol:
            break
        alpha = optimal_step(rosenbrock, x, d)
        x = x + (alpha * d)
        new_grad = gradient(x)
        beta = np.dot(new_grad, new_grad - grad) / np.dot(grad, grad)
        d = -new_grad + beta * d
        grad = new_grad
        trajectory.append(x.copy())
    
    return np.array(trajectory)

def plot_trajectories(traj_gd, traj_cg, traj_gdd):
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])
    
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
    plt.plot(traj_gd[:, 0], traj_gd[:, 1], 'ro-', label='Gradient Optimal Step Descent')

    plt.plot(traj_gdd[:, 0], traj_gdd[:, 1], 'go-', label='Gradient Decaying Step Descent')

    plt.plot(traj_cg[:, 0], traj_cg[:, 1], 'bo-', label='Conjugate Gradient')
    plt.legend()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Optimization Trajectories')
    plt.show()

x0 = [-1, -1]
traj_gd = gradient_descent_optimal_stepsize(x0)
traj_gdd = gradient_descent_decaying_stepsize(x0)
traj_cg = conjugate_gradient(x0)
plot_trajectories(traj_gd, traj_cg, traj_gdd)
