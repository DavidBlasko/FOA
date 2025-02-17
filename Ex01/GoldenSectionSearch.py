import numpy as np
import matplotlib.pyplot as plt

# function we search on
def f(x):
    return 0.2 * np.exp(x - 2) - x

# Golden Section Search
def golden_search(f, a, b, n):
    phi = (1 + np.sqrt(5)) / 2
    rho = (phi - 1)
    d = rho * b + (1 - rho) * a
    yd = f(d)
    
    for i in range(1, n):
        c = rho * a + (1 - rho) * b
        yc = f(c)
        
        if yc < yd:
            b = d
            d = c
            yd = yc
        else:
            a = b
            b = c

    return (a, b) if a < b else (b, a)

# graph func
def visualize_golden_search():
    a, b = -1, 5
    n = 5
    
    interval_start, interval_end = golden_search(f, a, b, n)
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
    plt.title('Final Bracket After 5 Golden Search Evaluations')
    plt.show()

# graph function call
visualize_golden_search()