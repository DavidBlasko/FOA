import numpy as np
import matplotlib.pyplot as plt

# function we search on
def f(x):
    return 0.2 * np.exp(x - 2) - x

# Fibonacci Search
def fibonacci_search(f, a, b, n, epsilon=0.01):
    phi = (1 + np.sqrt(5)) / 2
    s = (1 - np.sqrt(5)) / (1 + np.sqrt(5))
    rho = (1 - s ** n) / (phi * (1 - s ** (n + 1)))
    d = rho * b + (1 - rho) * a
    yd = f(d)
    
    for i in range(1, n):
        if i == n - 1:
            c = epsilon * a + (1 - epsilon) * d
        else:
            c = rho * a + (1 - rho) * b
        
        yc = f(c)
        
        if yc < yd:
            b = d
            d = c
            yd = yc
        else:
            a = b
            b = c
        
        rho = (1 - s ** (n - i)) / (phi * (1 - s ** (n - i + 1)))
    
    return (a, b) if a < b else (b, a)

# graph func
def visualize_fibonacci_search():
    a, b = -1, 5
    n = 5
    
    interval_start, interval_end = fibonacci_search(f, a, b, n)
    x = np.linspace(a, b, 100)
    y = f(x)
    final_interval_length = interval_end - interval_start
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label='Function f(x) = 0.2 * exp(x - 2) - x')

    # final bracket plot
    plt.axvline(interval_start, color='r', linestyle='--', label=f'Interval Start ({interval_start:.4f})')
    plt.axvline(interval_end, color='r', linestyle='--', label=f'Interval End ({interval_end:.4f})')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(title=f'Final interval length: {final_interval_length:.4f}')
    plt.title('Final Bracket After 5 Fibonacci Search Evaluations')
    plt.show()

# graph function call
visualize_fibonacci_search()