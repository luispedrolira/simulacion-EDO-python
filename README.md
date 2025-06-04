# Proyecto Final - Ecuaciones Diferenciales 1

## Descripción

Este proyecto es una simulación y comparación de métodos numéricos para la resolución de ecuaciones diferenciales ordinarias (EDO), desarrollado como trabajo final para el curso de **Ecuaciones Diferenciales 1**. El objetivo es resolver distintos tipos de EDOs utilizando los métodos de **Runge-Kutta de cuarto orden (RK4)** y **Adams-Bashforth de segundo orden**, y comparar sus resultados con las soluciones analíticas.

**Autores:**
- Juan Mario Padilla (23927)  
- Elena Sobalvarro (23991)  
- Luis Pedro Lira (23669)

---

## Problemas resueltos

El programa resuelve y grafica los siguientes casos:

### 1. EDO de primer orden

```
dy/dt = -2y + 1,    y(0) = 0
Solución analítica: y(t) = (1/2)(1 - e^(-2t))
```

---

### 2. EDO de segundo orden

```
d²y/dt² = -9y,    y(0) = 1,    y'(0) = 0
Solución analítica: y(t) = cos(3t)
```

---

### 3. Sistema de EDOs (2x2)

```
dx/dt = 3x + 4y
dy/dt = -4x + 3y
Condiciones iniciales: x(0) = 1, y(0) = 0

Solución analítica:
x(t) = e^(3t) * cos(4t)
y(t) = -e^(3t) * sin(4t)
```

---

Para cada caso, se grafican las soluciones numéricas y analíticas, y se calcula el error de cada método.

---

## Métodos numéricos implementados

- **Runge-Kutta de cuarto orden (RK4):**  
  Implementado para EDOs de primer orden y sistemas de EDOs.

- **Adams-Bashforth de segundo orden:**  
  Implementado para EDOs de primer orden y sistemas de EDOs. Utiliza RK4 para obtener el segundo punto inicial.

> Las implementaciones se encuentran en la carpeta `metodos_numericos/`.

---

## Requisitos

- Python 3.x  
- `numpy`  
- `matplotlib`

Puedes instalar las dependencias ejecutando:

```bash
pip install numpy matplotlib
```

---

## Ejecución

Para ejecutar el proyecto, simplemente corre el archivo principal:

```bash
python main.py
```

Se mostrarán las gráficas comparativas y los análisis de error en la terminal.

---

## Estructura del proyecto

```
proyectofinal/
│
├── main.py
├── metodos_numericos/
│   ├── rk4.py
│   ├── adams_bashforth.py
│   └── __init__.py
```

- `main.py`: Script principal que resuelve y grafica los problemas.  
- `metodos_numericos/`: Implementaciones de los métodos numéricos.

---

## Créditos

Proyecto realizado para el curso de **Ecuaciones Diferenciales 1**, Universidad del Valle de Guatemala (UVG).
