import numpy as np

# METODOS NUMERICOS ADAMS-BASHFORTH

def adams_bashforth_2(f, y0, t0, tf, h):
    """
    Método de Adams-Bashforth de segundo orden para EDO de primer orden
    Formula: y_{n+1} = y_n + (h/2)[3f(x_n, y_n) - f(x_{n-1}, y_{n-1})]
    Usa RK4 para calcular el segundo punto inicial
    """
    n = int((tf - t0) / h)
    t_values = np.linspace(t0, tf, n+1)
    y_values = np.zeros(n+1)
    y_values[0] = y0
    
    # Calcular y_1 usando RK4 (necesario para Adams-Bashforth)
    if n > 0:
        t = t_values[0]
        y = y_values[0]
        k1 = h * f(t, y)
        k2 = h * f(t + h/2, y + k1/2)
        k3 = h * f(t + h/2, y + k2/2)
        k4 = h * f(t + h, y + k3)
        y_values[1] = y + (k1 + 2*k2 + 2*k3 + k4)/6
    
    # Aplicar Adams-Bashforth para el resto de puntos
    for i in range(1, n):
        t_n = t_values[i]
        t_n_minus_1 = t_values[i-1]
        y_n = y_values[i]
        y_n_minus_1 = y_values[i-1]
        
        f_n = f(t_n, y_n)
        f_n_minus_1 = f(t_n_minus_1, y_n_minus_1)
        
        y_values[i+1] = y_n + (h/2) * (3*f_n - f_n_minus_1)
    
    return t_values, y_values



def adams_bashforth_2_sistema(f, u0, t0, tf, h):
    """Método de Adams-Bashforth de segundo orden para sistemas de EDOs"""
    n = int((tf - t0) / h)
    t_values = np.linspace(t0, tf, n+1)
    u_values = np.zeros((n+1, len(u0)))
    u_values[0] = u0
    
    # Calcular u_1 usando RK4
    if n > 0:
        t = t_values[0]
        u = u_values[0]
        k1 = h * f(t, u)
        k2 = h * f(t + h/2, u + k1/2)
        k3 = h * f(t + h/2, u + k2/2)
        k4 = h * f(t + h, u + k3)
        u_values[1] = u + (k1 + 2*k2 + 2*k3 + k4)/6
    
    # Aplicar Adams-Bashforth
    for i in range(1, n):
        t_n = t_values[i]
        t_n_minus_1 = t_values[i-1]
        u_n = u_values[i]
        u_n_minus_1 = u_values[i-1]
        
        f_n = f(t_n, u_n)
        f_n_minus_1 = f(t_n_minus_1, u_n_minus_1)
        
        u_values[i+1] = u_n + (h/2) * (3*f_n - f_n_minus_1)
    
    return t_values, u_values