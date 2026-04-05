"""Лабораторная работа: Численные вычисления и анализ данных с NumPy."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. СОЗДАНИЕ И ОБРАБОТКА МАССИВОВ
# ============================================================

def create_vector() -> np.ndarray:
    """Создать массив от 0 до 9 включительно.

    Returns:
        np.ndarray: Массив чисел от 0 до 9.
    """
    return np.arange(10)


def create_matrix() -> np.ndarray:
    """Создать матрицу 5x5 со случайными числами из [0, 1).

    Returns:
        np.ndarray: Матрица 5x5 со случайными значениями.
    """
    return np.random.rand(5, 5)


def reshape_vector(vec: np.ndarray) -> np.ndarray:
    """Преобразовать вектор формы (10,) в матрицу формы (2, 5).

    Args:
        vec: Входной массив формы (10,).

    Returns:
        np.ndarray: Массив формы (2, 5).
    """
    return vec.reshape(2, 5)


def transpose_matrix(mat: np.ndarray) -> np.ndarray:
    """Транспонировать матрицу.

    Args:
        mat: Входная матрица.

    Returns:
        np.ndarray: Транспонированная матрица.
    """
    return mat.T


# ============================================================
# 2. ВЕКТОРНЫЕ ОПЕРАЦИИ
# ============================================================

def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Поэлементное сложение двух векторов.

    Args:
        a: Первый вектор.
        b: Второй вектор.

    Returns:
        np.ndarray: Результат сложения.
    """
    return a + b


def scalar_multiply(vec: np.ndarray, scalar: float) -> np.ndarray:
    """Умножение вектора на скаляр.

    Args:
        vec: Входной вектор.
        scalar: Число-множитель.

    Returns:
        np.ndarray: Результат умножения.
    """
    return vec * scalar


def elementwise_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Поэлементное умножение двух массивов.

    Args:
        a: Первый массив.
        b: Второй массив.

    Returns:
        np.ndarray: Результат поэлементного умножения.
    """
    return a * b


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Скалярное произведение двух векторов.

    Args:
        a: Первый вектор.
        b: Второй вектор.

    Returns:
        float: Скалярное произведение.
    """
    return np.dot(a, b)


# ============================================================
# 3. МАТРИЧНЫЕ ОПЕРАЦИИ
# ============================================================

def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Умножение двух матриц.

    Args:
        a: Первая матрица.
        b: Вторая матрица.

    Returns:
        np.ndarray: Результат матричного умножения.
    """
    return a @ b


def matrix_determinant(a: np.ndarray) -> float:
    """Вычислить определитель квадратной матрицы.

    Args:
        a: Квадратная матрица.

    Returns:
        float: Определитель матрицы.
    """
    return np.linalg.det(a)


def matrix_inverse(a: np.ndarray) -> np.ndarray:
    """Вычислить обратную матрицу.

    Args:
        a: Квадратная невырожденная матрица.

    Returns:
        np.ndarray: Обратная матрица.
    """
    return np.linalg.inv(a)


def solve_linear_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Решить систему линейных уравнений Ax = b.

    Args:
        a: Матрица коэффициентов A.
        b: Вектор свободных членов b.

    Returns:
        np.ndarray: Вектор решений x.
    """
    return np.linalg.solve(a, b)


# ============================================================
# 4. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================

def load_dataset(path: str = "data/students_scores.csv") -> np.ndarray:
    """Загрузить CSV-файл и вернуть NumPy-массив.

    Args:
        path: Путь к CSV-файлу.

    Returns:
        np.ndarray: Данные из файла.
    """
    return pd.read_csv(path).to_numpy()


def statistical_analysis(data: np.ndarray) -> dict:
    """Вычислить основные статистические показатели.

    Args:
        data: Одномерный массив числовых данных.

    Returns:
        dict: Словарь с ключами mean, median, std, min, max,
              percentile_25, percentile_75.
    """
    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "percentile_25": np.percentile(data, 25),
        "percentile_75": np.percentile(data, 75),
    }


def normalize_data(data: np.ndarray) -> np.ndarray:
    """Min-Max нормализация в диапазон [0, 1].

    Формула: (x - min) / (max - min).

    Args:
        data: Входной массив данных.

    Returns:
        np.ndarray: Нормализованный массив.
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_histogram(data: np.ndarray) -> None:
    plt.figure()
    plt.hist(data, bins=5, color="cyan", edgecolor="black")
    plt.title("Распределение оценок по математике")
    plt.xlabel("Оценки")
    plt.ylabel("Количество")
    plt.savefig("plots/hist.png")
    plt.show()


def plot_heatmap(matrix: np.ndarray) -> None:
    plt.figure()
    sns.heatmap(matrix, annot=True, square=True)
    plt.title("Тепловая карта корреляции предметов")
    plt.savefig("plots/heatmap.png")
    plt.show()


def plot_line(x: np.ndarray, y: np.ndarray) -> None:
    plt.figure()
    plt.plot(x, y, marker="o", linestyle="-", color="b")
    plt.title("Зависимость оценок от номера студента")
    plt.xlabel("Номер студента")
    plt.ylabel("Оценка по математике")
    plt.grid(True)
    plt.savefig("plots/line.png")
    plt.show()