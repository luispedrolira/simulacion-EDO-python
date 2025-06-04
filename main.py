from metodos_numericos.rk4 import runge_kutta_4, runge_kutta_4_sistema
from metodos_numericos.adams_bashforth import adams_bashforth_2, adams_bashforth_2_sistema
import numpy as np
import matplotlib.pyplot as plt

# |--------------------------------------------------------------------------|
# | PROYECTO FINAL: ECUACIONES DIFERENCIALES 1                               |
# | Juan Mario Padilla 23927, Elena Sobalvarro 23991, Luis Pedro Lira 23669  |
# | Alternativa 2: Simulación mediante métodos numéricos                     |
# |--------------------------------------------------------------------------|



# EDO DE PRIMER ORDEN
# donde la ecuaciones que escogimos fue: dy/dt = -2y + 1, y(0) = 0
# Y su solución analítica es: y(t) = (1/2)(1 - e^(-2t))
def f1(t, y):
    """Función para la EDO: dy/dt = -2y + 1"""
    return -2 * y + 1

def solucion_analitica_1(t):
    """Solución analítica: y(t) = (1/2)(1 - e^(-2t))"""
    return 0.5 * (1 - np.exp(-2 * t))

print("=" * 80)
print("PROBLEMA 1: EDO DE PRIMER ORDEN")
print("dy/dt = -2y + 1, y(0) = 0")
print("=" * 80)

# Parámetros
y0_1 = 0
t0_1 = 0
tf_1 = 2
h_1 = 0.1

# Aplicamos los métodos numéricos
t_rk4_1, y_rk4_1 = runge_kutta_4(f1, y0_1, t0_1, tf_1, h_1)
t_ab_1, y_ab_1 = adams_bashforth_2(f1, y0_1, t0_1, tf_1, h_1)

# Solución analítica
t_analitica_1 = np.linspace(t0_1, tf_1, 1000)
y_analitica_1 = solucion_analitica_1(t_analitica_1)

# Gráfica
plt.figure(figsize=(12, 8))
plt.plot(t_rk4_1, y_rk4_1, 'bs-', markersize=4, linewidth=2, label='Método RK4')
plt.plot(t_ab_1, y_ab_1, 'g^-', markersize=4, linewidth=2, label='Adams-Bashforth')
plt.plot(t_analitica_1, y_analitica_1, 'k-', linewidth=3, label='Solución Analítica')
plt.xlabel('t', fontsize=12)
plt.ylabel('y(t)', fontsize=12)
plt.title('EDO de Primer Orden: dy/dt = -2y + 1, y(0) = 0', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

# Análisis de error
valor_analitico_1 = solucion_analitica_1(tf_1)
error_rk4_1 = abs(y_rk4_1[-1] - valor_analitico_1)
error_ab_1 = abs(y_ab_1[-1] - valor_analitico_1)

print(f"Valor analítico en t={tf_1}: {valor_analitico_1:.8f}")
print(f"RK4:  {y_rk4_1[-1]:.8f}, Error: {error_rk4_1:.2e}")
print(f"Adams-Bashforth:  {y_ab_1[-1]:.8f}, Error: {error_ab_1:.2e}")




# EDO DE SEGUNDO ORDEN
# Donde la ecuaciones que escogimos fue: d²y/dt² = -9y, y(0) = 1, y'(0) = 0
# Y su solución analítica es: y(t) = cos(3t)
def sistema_segundo_orden_2(t, Y):
    """
    Convierte EDO de segundo orden en sistema de primer orden
    d²y/dt² = -9y se convierte en:
    y₁ = y, y₂ = y'
    dy₁/dt = y₂
    dy₂/dt = -9y₁
    """
    y1, y2 = Y  # y1 = y, y2 = y'
    dy1dt = y2
    dy2dt = -9 * y1
    return np.array([dy1dt, dy2dt])

def solucion_analitica_2(t):
    """Solución analítica: y(t) = cos(3t)"""
    return np.cos(3 * t)

print("\n" + "=" * 80)
print("PROBLEMA 2: EDO DE SEGUNDO ORDEN")
print("d²y/dt² = -9y, y(0) = 1, y'(0) = 0")
print("=" * 80)

# Parámetros
Y0_2 = np.array([1.0, 0.0])  # y(0) = 1, y'(0) = 0
t0_2 = 0
tf_2 = 2
h_2 = 0.05

# Aplicamos los métodos numéricos
t_rk4_2, Y_rk4_2 = runge_kutta_4_sistema(sistema_segundo_orden_2, Y0_2, t0_2, tf_2, h_2)
t_ab_2, Y_ab_2 = adams_bashforth_2_sistema(sistema_segundo_orden_2, Y0_2, t0_2, tf_2, h_2)

# Solución analítica
t_analitica_2 = np.linspace(t0_2, tf_2, 1000)
y_analitica_2 = solucion_analitica_2(t_analitica_2)

# Gráfica
plt.figure(figsize=(12, 8))
plt.plot(t_rk4_2, Y_rk4_2[:, 0], 'bs-', markersize=4, linewidth=2, label='Método RK4')
plt.plot(t_ab_2, Y_ab_2[:, 0], 'g^-', markersize=4, linewidth=2, label='Adams-Bashforth')
plt.plot(t_analitica_2, y_analitica_2, 'k-', linewidth=3, label='Solución Analítica')
plt.xlabel('t', fontsize=12)
plt.ylabel('y(t)', fontsize=12)
plt.title('EDO de Segundo Orden: d²y/dt² = -9y, y(0) = 1, y\'(0) = 0', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

# Análisis de error
valor_analitico_2 = solucion_analitica_2(tf_2)
error_rk4_2 = abs(Y_rk4_2[-1, 0] - valor_analitico_2)
error_ab_2 = abs(Y_ab_2[-1, 0] - valor_analitico_2)

print(f"Valor analítico en t={tf_2}: {valor_analitico_2:.8f}")
print(f"RK4:  {Y_rk4_2[-1, 0]:.8f}, Error: {error_rk4_2:.2e}")
print(f"A-B:  {Y_ab_2[-1, 0]:.8f}, Error: {error_ab_2:.2e}")




# SISTEMA DE EDOs 2x2
# Donde la ecuaciones que escogimos fue: dx/dt = 3x + 4y, dy/dt = -4x + 3y, x(0) = 1, y(0) = 0
# Y su solución analítica es: x(t) = e^(3t)cos(4t), y(t) = -e^(3t)sin(4t)
def sistema_2x2(t, u):
    """Sistema 2x2: dx/dt = 3x + 4y, dy/dt = -4x + 3y"""
    x, y = u
    dxdt = 3 * x + 4 * y
    dydt = -4 * x + 3 * y
    return np.array([dxdt, dydt])

def solucion_analitica_3(t):
    """Solución analítica del sistema 2x2"""
    x_analitica = np.exp(3 * t) * np.cos(4 * t)
    y_analitica = -np.exp(3 * t) * np.sin(4 * t)
    return x_analitica, y_analitica

print("\n" + "=" * 80)
print("PROBLEMA 3: SISTEMA DE EDOs 2x2")
print("dx/dt = 3x + 4y, dy/dt = -4x + 3y, x(0) = 1, y(0) = 0")
print("=" * 80)

# Parámetros
u0_3 = np.array([1.0, 0.0])  # x(0) = 1, y(0) = 0
t0_3 = 0
tf_3 = 1
h_3 = 0.02

# Aplicamos los métodos numéricos
t_rk4_3, u_rk4_3 = runge_kutta_4_sistema(sistema_2x2, u0_3, t0_3, tf_3, h_3)
t_ab_3, u_ab_3 = adams_bashforth_2_sistema(sistema_2x2, u0_3, t0_3, tf_3, h_3)

# Solución analítica
t_analitica_3 = np.linspace(t0_3, tf_3, 1000)
x_analitica_3, y_analitica_3 = solucion_analitica_3(t_analitica_3)

# Gráfica para la componente x(t)
plt.figure(figsize=(12, 8))
plt.plot(t_rk4_3, u_rk4_3[:, 0], 'bs-', markersize=4, linewidth=2, label='Método RK4')
plt.plot(t_ab_3, u_ab_3[:, 0], 'g^-', markersize=4, linewidth=2, label='Adams-Bashforth')
plt.plot(t_analitica_3, x_analitica_3, 'k-', linewidth=3, label='Solución Analítica')
plt.xlabel('t', fontsize=12)
plt.ylabel('x(t)', fontsize=12)
plt.title('Sistema 2x2 - Componente x(t)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

# Gráfica para la componente y(t)
plt.figure(figsize=(12, 8))
plt.plot(t_rk4_3, u_rk4_3[:, 1], 'bs-', markersize=4, linewidth=2, label='Método RK4')
plt.plot(t_ab_3, u_ab_3[:, 1], 'g^-', markersize=4, linewidth=2, label='Adams-Bashforth')
plt.plot(t_analitica_3, y_analitica_3, 'k-', linewidth=3, label='Solución Analítica')
plt.xlabel('t', fontsize=12)
plt.ylabel('y(t)', fontsize=12)
plt.title('Sistema 2x2 - Componente y(t)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

# Análisis de error
x_analitico_final, y_analitico_final = solucion_analitica_3(tf_3)

error_rk4_3x = abs(u_rk4_3[-1, 0] - x_analitico_final)
error_ab_3x = abs(u_ab_3[-1, 0] - x_analitico_final)
error_rk4_3y = abs(u_rk4_3[-1, 1] - y_analitico_final)
error_ab_3y = abs(u_ab_3[-1, 1] - y_analitico_final)

print(f"COMPONENTE X en t={tf_3}:")
print(f"Valor analítico: {x_analitico_final:.8f}")
print(f"RK4:  {u_rk4_3[-1, 0]:.8f}, Error: {error_rk4_3x:.2e}")
print(f"A-B:  {u_ab_3[-1, 0]:.8f}, Error: {error_ab_3x:.2e}")

print(f"\nCOMPONENTE Y en t={tf_3}:")
print(f"Valor analítico: {y_analitico_final:.8f}")
print(f"RK4:  {u_rk4_3[-1, 1]:.8f}, Error: {error_rk4_3y:.2e}")
print(f"A-B:  {u_ab_3[-1, 1]:.8f}, Error: {error_ab_3y:.2e}")


# RESUMEN COMPARATIVO DE MÉTODOS
print("\n" + "=" * 80)
print("RESUMEN COMPARATIVO DE PRECISIÓN")
print("=" * 80)

print("\nOrden de precisión por problema:")
print("1. EDO Primer Orden:")
errores_1 = [("RK4", error_rk4_1), ("Adams-Bashforth", error_ab_1)]
errores_1.sort(key=lambda x: x[1])
for i, (metodo, error) in enumerate(errores_1, 1):
    print(f"   {i}. {metodo}: {error:.2e}")

print("\n2. EDO Segundo Orden:")
errores_2 = [("RK4", error_rk4_2), ("Adams-Bashforth", error_ab_2)]
errores_2.sort(key=lambda x: x[1])
for i, (metodo, error) in enumerate(errores_2, 1):
    print(f"   {i}. {metodo}: {error:.2e}")

print("\n3. Sistema 2x2 (componente x):")
errores_3x = [("RK4", error_rk4_3x), ("Adams-Bashforth", error_ab_3x)]
errores_3x.sort(key=lambda x: x[1])
for i, (metodo, error) in enumerate(errores_3x, 1):
    print(f"   {i}. {metodo}: {error:.2e}")

print("\nConclusión:")
print("- RK4 generalmente proporciona la mayor precisión")
print("- Adams-Bashforth es eficiente para integraciones de largo plazo")
print("=" * 80)