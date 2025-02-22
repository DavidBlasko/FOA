import numpy as np
import matplotlib.pyplot as plt

# function we search on
def f(x):
    return 0.2 * np.exp(x - 2) - x

# Quadratic Fit Search
def quadratic_fit_search(f, a, b, c, n):
    ya, yb, yc = f(a), f(b), f(c)
    
    for i in range(1, n - 2):
        numerator = ya * (b**2 - c**2) + yb * (c**2 - a**2) + yc * (a**2 - b**2)
        denominator = ya * (b - c) + yb * (c - a) + yc * (a - b)
        x = 1 / 2 * (numerator / denominator)
        yx = f(x)
        
        if x > b:
            if yx > yb:
                c, yc = x, yx
            else:
                a, ya, b, yb = b, yb, x, yx
        else:
            if yx > yb:
                a, ya = x, yx
            else:
                c, yc, b, yb = b, yb, x, yx

    return a, b, c

# graph func
def visualize_quadratic_fit_search():
    a, b, c = -1, 2, 5
    n = 5
    
    interval_start, last_b, interval_end = quadratic_fit_search(f, a, b, c, n)
    x = np.linspace(a, c, 100)
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
    plt.title('Final Bracket After 5 Quadratic Fit Search Evaluations')
    plt.show()

# graph function call
visualize_quadratic_fit_search()