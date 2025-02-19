clc; clear;

f = @(x) 0.2*exp(x-2)-x;
n = 5;
a = -1;
a_b = a;
b = 5;
b_b = b;
eps = 0.001;
phi = (1+sqrt(5))/2;

s = (1-sqrt(5))/(1+sqrt(5));
rho = 1/(phi*(1-s^(n+1))/(1+s^n));
d = rho*b+(1-rho)*a;
yd = f(d);

for i = 1:n-1
    if i == n-1
        c = eps*a+(1-eps)*d;
    else
        c = rho*a+(1-rho)*b;
    end
    yc = f(c);

    if yc < yd
        b = d; d = c; yd = yc;
    else
        a = b; b = c;
    end
    rho = (1-s^(n-i))/(phi*(1-s^(n+1-i)));
end

if a < b
    return;
else
    temp = a; a = b; b = temp;
end

hold on;
fplot(f, [-5, 5])
plot([a_b, b_b], [f(a_b), f(b_b)], 'bx', 'LineWidth', 2)
plot([a, b], [f(a), f(b)], 'rx', 'LineWidth', 2)