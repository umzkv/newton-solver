"""
Newton's Method for Non-Linear Systems of Equations
=====================================================
Метод Ньютона для систем нелинейных уравнений.

Метод решает систему F(x) = 0, итерируя по формуле:
    x_{k+1} = x_k - J(x_k)^{-1} * F(x_k)

где J(x_k) — матрица Якоби в точке x_k.
"""

import numpy as np
import sys


# ─────────────────────────────────────────────
#  Ядро метода Ньютона
# ─────────────────────────────────────────────

def numerical_jacobian(F, x, eps=1e-7):
    """
    Вычисляет матрицу Якоби численно методом конечных разностей.

    Parameters
    ----------
    F   : callable — вектор-функция F(x), возвращает np.ndarray размера n
    x   : np.ndarray — текущая точка, размер n
    eps : float — шаг для численного дифференцирования

    Returns
    -------
    J : np.ndarray (n x n) — матрица Якоби
    """
    n = len(x)
    J = np.zeros((n, n))
    f0 = F(x)

    for j in range(n):
        x_shifted = x.copy()
        x_shifted[j] += eps
        J[:, j] = (F(x_shifted) - f0) / eps

    return J


def newton_method(F, x0, tol=1e-10, max_iter=100, verbose=True):
    """
    Метод Ньютона для систем нелинейных уравнений.

    Parameters
    ----------
    F        : callable — вектор-функция F(x) = 0, возвращает np.ndarray
    x0       : array-like — начальное приближение
    tol      : float — порог сходимости (норма шага ||Δx||)
    max_iter : int — максимальное число итераций
    verbose  : bool — выводить ли таблицу итераций

    Returns
    -------
    x        : np.ndarray — найденное решение
    info     : dict — информация о сходимости:
                 'iterations' — количество итераций
                 'residual'   — ||F(x)|| в финальной точке
                 'converged'  — True / False
                 'history'    — список (iter, x, ||F(x)||) по каждой итерации
    """
    x = np.array(x0, dtype=float)
    history = []

    if verbose:
        print(f"\n{'─'*65}")
        print(f"  Метод Ньютона   |  tol = {tol}  |  max_iter = {max_iter}")
        print(f"{'─'*65}")
        print(f"  {'Iter':>4}  {'||F(x)||':>14}  {'||Δx||':>14}  x")
        print(f"{'─'*65}")

    for k in range(max_iter):
        Fx = F(x)
        residual = np.linalg.norm(Fx)
        history.append((k, x.copy(), residual))

        J = numerical_jacobian(F, x)

        # Решаем линейную систему J * Δx = -F(x)
        try:
            delta_x = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            print("  ✗ Якобиан вырожден — метод не сошёлся.")
            return x, {
                "iterations": k,
                "residual": residual,
                "converged": False,
                "history": history,
            }

        x = x + delta_x
        step_norm = np.linalg.norm(delta_x)

        if verbose:
            x_str = np.array2string(x, precision=8, separator=", ")
            print(f"  {k+1:>4}  {residual:>14.6e}  {step_norm:>14.6e}  {x_str}")

        if step_norm < tol:
            final_residual = np.linalg.norm(F(x))
            if verbose:
                print(f"{'─'*65}")
                print(f"  ✓ Сошлось за {k+1} итераций")
                print(f"  Решение   : {np.array2string(x, precision=10, separator=', ')}")
                print(f"  ||F(x)||  : {final_residual:.6e}")
                print(f"{'─'*65}\n")
            return x, {
                "iterations": k + 1,
                "residual": final_residual,
                "converged": True,
                "history": history,
            }

    final_residual = np.linalg.norm(F(x))
    if verbose:
        print(f"{'─'*65}")
        print(f"  ✗ Достигнут лимит итераций ({max_iter}). Решение приближённое.")
        print(f"  Решение   : {np.array2string(x, precision=10, separator=', ')}")
        print(f"  ||F(x)||  : {final_residual:.6e}")
        print(f"{'─'*65}\n")

    return x, {
        "iterations": max_iter,
        "residual": final_residual,
        "converged": False,
        "history": history,
    }


# ─────────────────────────────────────────────
#  Примеры систем уравнений
# ─────────────────────────────────────────────

def example_1():
    """
    Система 2x2:
        x² + y² - 4  = 0   (окружность радиуса 2)
        x² - y  - 1  = 0   (парабола)

    Одно из решений: x ≈ 1.2361, y ≈ 0.5279
    """
    def F(v):
        x, y = v
        return np.array([
            x**2 + y**2 - 4,
            x**2 - y - 1,
        ])

    print("\n" + "═"*65)
    print("  ПРИМЕР 1: окружность ∩ парабола (2×2)")
    print("  x² + y² - 4 = 0")
    print("  x² - y - 1  = 0")
    print("═"*65)

    x0 = [1.0, 0.5]
    solution, info = newton_method(F, x0)
    verify(F, solution)


def example_2():
    """
    Система 3x3:
        3x₁ - cos(x₂·x₃) - 0.5    = 0
        x₁² - 81(x₂+0.1)² + sin(x₃) + 1.06 = 0
        e^(-x₁·x₂) + 20x₃ + (10π-3)/3      = 0

    Классический тестовый пример (Burden & Faires).
    Точное решение: x₁=0.5, x₂=0, x₃=-0.5236 (≈ -π/6)
    """
    def F(v):
        x1, x2, x3 = v
        return np.array([
            3*x1 - np.cos(x2*x3) - 0.5,
            x1**2 - 81*(x2 + 0.1)**2 + np.sin(x3) + 1.06,
            np.exp(-x1*x2) + 20*x3 + (10*np.pi - 3) / 3,
        ])

    print("\n" + "═"*65)
    print("  ПРИМЕР 2: классическая система 3×3 (Burden & Faires)")
    print("  3x₁ - cos(x₂x₃) - 0.5 = 0")
    print("  x₁² - 81(x₂+0.1)² + sin(x₃) + 1.06 = 0")
    print("  e^(-x₁x₂) + 20x₃ + (10π-3)/3 = 0")
    print("═"*65)

    x0 = [0.1, 0.1, -0.1]
    solution, info = newton_method(F, x0)
    verify(F, solution)


def example_3():
    """
    Система 2x2 (два пересечения двух нелинейных функций):
        sin(x) + y  - 1.2 = 0
        x + cos(y) - 0.8  = 0
    """
    def F(v):
        x, y = v
        return np.array([
            np.sin(x) + y - 1.2,
            x + np.cos(y) - 0.8,
        ])

    print("\n" + "═"*65)
    print("  ПРИМЕР 3: тригонометрическая система 2×2")
    print("  sin(x) + y  - 1.2 = 0")
    print("  x + cos(y) - 0.8  = 0")
    print("═"*65)

    x0 = [0.5, 0.5]
    solution, info = newton_method(F, x0)
    verify(F, solution)


def verify(F, x):
    """Проверяет, что F(x) ≈ 0."""
    Fx = F(x)
    print(f"  Проверка F(x) = {np.array2string(Fx, precision=4, separator=', ')}\n")


# ─────────────────────────────────────────────
#  Интерактивный ввод пользователя
# ─────────────────────────────────────────────

def interactive_mode():
    """
    Позволяет пользователю ввести собственную систему уравнений в виде
    Python-выражений и начальное приближение.
    """
    print("\n" + "═"*65)
    print("  ИНТЕРАКТИВНЫЙ РЕЖИМ — введите свою систему уравнений")
    print("═"*65)
    print("  Введите уравнения вида F_i(x) = 0.")
    print("  Переменные: x[0], x[1], x[2], ...")
    print("  Доступные функции: np.sin, np.cos, np.exp, np.log, ...")
    print()

    n = int(input("  Количество уравнений (= количество переменных): "))
    equations = []
    print(f"  Введите {n} выражений (левая часть уравнения = 0):")
    for i in range(n):
        expr = input(f"    F[{i}] = ").strip()
        equations.append(expr)

    print(f"\n  Введите начальное приближение x0 ({n} чисел через пробел):")
    x0 = list(map(float, input("  x0 = ").split()))

    print("\n  Параметры метода (Enter — значение по умолчанию):")
    tol_input = input("  Точность tol [1e-10]: ").strip()
    tol = float(tol_input) if tol_input else 1e-10
    max_iter_input = input("  Макс. итераций [100]: ").strip()
    max_iter = int(max_iter_input) if max_iter_input else 100

    # Компилируем пользовательские уравнения
    def F(x_vec):
        x = x_vec  # alias for user expressions
        result = np.zeros(n)
        for i, expr in enumerate(equations):
            result[i] = eval(expr, {"np": np, "x": x_vec, "__builtins__": {}})
        return result

    print("\n  Запуск метода Ньютона...")
    solution, info = newton_method(F, x0, tol=tol, max_iter=max_iter)
    verify(F, solution)


# ─────────────────────────────────────────────
#  Точка входа
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "╔" + "═"*63 + "╗")
    print("║" + "  Newton's Method — нелинейные системы уравнений".center(63) + "║")
    print("╚" + "═"*63 + "╝")

    print("""
  Выберите режим:
    1 — Пример 1: окружность ∩ парабола         (2×2)
    2 — Пример 2: классическая система           (3×3)
    3 — Пример 3: тригонометрическая система     (2×2)
    4 — Интерактивный ввод своей системы
    0 — Запустить все примеры
""")

    choice = input("  Ваш выбор: ").strip()

    if choice == "1":
        example_1()
    elif choice == "2":
        example_2()
    elif choice == "3":
        example_3()
    elif choice == "4":
        interactive_mode()
    elif choice == "0":
        example_1()
        example_2()
        example_3()
    else:
        print("  Неверный выбор. Запускаем все примеры...")
        example_1()
        example_2()
        example_3()
