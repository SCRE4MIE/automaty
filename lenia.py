import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import matplotlib.animation as animation
import matplotlib
import multiprocessing
from scipy.ndimage import gaussian_filter
matplotlib.use('TkAgg')

# --------------- loading points to matrix -------------------------
def load_points(matrix, points_x: list, points_y: list):
    if len(points_x) != len(points_y):
        raise Exception('Lists are not eaqual!')
    for i in range(len(points_x)):
        matrix[points_y[i]][points_x[i]] = 1


def load_file(matrix, file):
    """Ładuje plik z danymi."""

    lista = []
    with open(file, 'r') as file:
        for line in file:
            lista.append(list(map(lambda e: float(e), line.replace('\n', '').split())))
    for i in range(len(lista)):
        matrix[int(lista[i][1])][int(lista[i][0])] = 1


# -------------------------------------------------------------------
def create_kernel(outer_radius, inner_radius, smoothing_factor):
    size = 2 * outer_radius + 1
    x, y = np.meshgrid(np.arange(size) - outer_radius, np.arange(size) - outer_radius)

    distance = np.sqrt(x ** 2 + y ** 2)

    # Create a binary ring mask by subtracting the inner circle from the outer circle
    outer_circle = (distance <= outer_radius).astype(float)
    inner_circle = (distance <= inner_radius).astype(float)
    ring = outer_circle - inner_circle

    # Apply Gaussian smoothing to the ring
    smoothed_ring = gaussian_filter(ring, sigma=smoothing_factor)

    return smoothed_ring



