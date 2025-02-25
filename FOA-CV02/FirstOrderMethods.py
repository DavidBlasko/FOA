import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x, a=1, b=5):
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2

def gradient(x, a=1, b=5):
    dfdx1 = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0]**2)
    dfdx2 = 2 * b * (x[1] - x[0]**2)
    return np.array([dfdx1, dfdx2])

def golden_section_search(f, x, d, a=0, b=1, tol=1e-5):
    phi = (1 + np.sqrt(5)) / 2  # golden search
    resphi = 2 - phi
    
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    f1 = f(x + x1 * d)
    f2 = f(x + x2 * d)
    
    while abs(b - a) > tol:
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = a + resphi * (b - a)
            f1 = f(x + x1 * d)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = b - resphi * (b - a)
            f2 = f(x + x2 * d)
    
    return (a + b) / 2

def gradient_descent_optimal_stepsize(x0, max_iter=100, tol=1e-2): # optimal step size with golden search
    x = np.array(x0)
    trajectory = [x.copy()]
    
    for k in range(max_iter):
        grad = gradient(x)
        if np.linalg.norm(grad) < tol:
            break
        alpha = golden_section_search(rosenbrock, x, -grad)  # optimal step with golden search
        x = x - (alpha * grad)
        trajectory.append(x.copy())
    
    return np.array(trajectory)

def gradient_descent_decaying_stepsize(x0, max_iter=100, tol=1e-2): # decaying step size
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

def conjugate_gradient(x0, max_iter=100, tol=1e-2): # conjugate gradient method
    x = np.array(x0)
    trajectory = [x.copy()]
    grad = gradient(x)
    d = -grad
    iter_count = 0
    
    for k in range(max_iter):
        if np.linalg.norm(grad) < tol:
            break
        alpha = golden_section_search(rosenbrock, x, d)
        x = x + (alpha * d)
        new_grad = gradient(x)
        beta = np.dot(new_grad, new_grad - grad) / np.dot(grad, grad)
        d = - new_grad + beta * d
        grad = new_grad
        trajectory.append(x.copy())
        iter_count = k + 1
    
    return np.array(trajectory), iter_count

def plot_trajectories(traj_gdo, traj_gdd, traj_cg, cg_iters):
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])
    
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='jet')
    plt.plot(traj_gdo[:, 0], traj_gdo[:, 1], 'ro-', label='Gradient Optimal Step Descent')
    plt.plot(traj_gdd[:, 0], traj_gdd[:, 1], 'go-', label='Gradient Decaying Step Descent')
    plt.plot(traj_cg[:, 0], traj_cg[:, 1], 'bo-', label=f'Conjugate Gradient ({cg_iters} iterations)')
    plt.legend()
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title('Optimization Trajectories')
    plt.show()

x0 = [-1, -1]
traj_gdo = gradient_descent_optimal_stepsize(x0)
traj_gdd = gradient_descent_decaying_stepsize(x0)
traj_cg, cg_iters = conjugate_gradient(x0)
plot_trajectories(traj_gdo, traj_gdd, traj_cg, cg_iters)