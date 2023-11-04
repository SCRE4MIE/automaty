import matplotlib.pyplot as plt
import numpy as np
import random
import copy
from matplotlib.animation import FuncAnimation
import matplotlib
import multiprocessing
from scipy.ndimage import gaussian_filter

# matplotlib.use('macosx')
matplotlib.use('TkAgg')
# matplotlib.use("Agg")

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

def calc_U(matrix, i_c, j_c, kernel, count_k, shape_kernel, half_size_i_kernel, half_size_j_kernel, n, m):
    u = 0
    for i_k in range(shape_kernel[0]):
        for j_k in range(shape_kernel[1]):
            i_matrix_index = i_c - half_size_i_kernel + i_k
            j_matrix_index = j_c - half_size_j_kernel + j_k
            u += matrix[i_matrix_index % n][j_matrix_index % m] * kernel[i_k][j_k]
    u = u / count_k
    return u


def growth_func(u, sigma, mu):
    l = abs(u - mu)
    k = 2 * (sigma ** 2)
    return 2 * np.exp(-(l ** 2) / k) - 1


def calc_c_t(matrix, i, j, at, sigma, mu, kernel, count_k, shape_kernel, half_size_i_kernel, half_size_j_kernel, n, m):
    u = calc_U(matrix=matrix, i_c=i, j_c=j, kernel=kernel, count_k=count_k, shape_kernel=shape_kernel, half_size_i_kernel=half_size_i_kernel, half_size_j_kernel=half_size_j_kernel, n=n, m=m)
    a = growth_func(u, sigma=sigma, mu=mu)
    return np.clip((matrix[i][j] + at * a), 0, 1)


def matrix_loop_parallel(start_i, end_i, start_j, end_j, matrix, at, sigma, mu, kernel, count_k, shape_kernel, half_size_i_kernel, half_size_j_kernel, n, m, matrix_tmp, pipe):
    for i_o in range(start_i, end_i):
        for j_o in range(start_j, end_j):
            matrix_tmp[i_o][j_o] = calc_c_t(matrix=matrix, i=i_o, j=j_o, at=at, sigma=sigma, mu=mu, kernel=kernel, count_k=count_k, shape_kernel=shape_kernel,
                                            half_size_i_kernel=half_size_i_kernel, half_size_j_kernel=half_size_j_kernel, n=n, m=m)

    pipe.send(matrix_tmp)


if __name__ == '__main__':

    n = 200
    m = 200
    kernel_outer_radius = 12
    kernel_inner_radius = 7
    smoothing_factor = 1.5
    at = 0.1
    sigma = 0.03
    mu = 0.23

    kernel = create_kernel(kernel_outer_radius, kernel_inner_radius, smoothing_factor=smoothing_factor)
    matrix = np.zeros((n, m))

    count_k = np.sum(kernel)
    shape_kernel = len(kernel), len(kernel[0])
    half_size_i_kernel = int(shape_kernel[0] / 2)
    half_size_j_kernel = int(shape_kernel[1] / 2)

    #  -------- LOAD POINTS-------
    load_points(matrix=matrix, points_x=[random.randint(30, 90) for _ in range(5000)],
                      points_y=[random.randint(30, 90) for _ in range(5000)])
    load_points(matrix=matrix, points_x=[random.randint(60, 90) for _ in range(5000)],
                points_y=[random.randint(60, 90) for _ in range(5000)])
    load_points(matrix=matrix, points_x=[random.randint(90, 120) for _ in range(5000)],
                points_y=[random.randint(90, 120) for _ in range(5000)])
    load_points(matrix=matrix, points_x=[random.randint(0, 199) for _ in range(10000)],
                points_y=[random.randint(0, 199) for _ in range(10000)])
    # ----------------------------

    fig = plt.figure(figsize=(8, 8))
    im = plt.imshow(matrix, cmap='jet', animated=True)
    plt.axis('off')
    count = 0

    # --- PIPE ------------------
    parent_1, child_1 = multiprocessing.Pipe()
    parent_2, child_2 = multiprocessing.Pipe()
    parent_3, child_3 = multiprocessing.Pipe()
    parent_4, child_4 = multiprocessing.Pipe()
    parent_5, child_5 = multiprocessing.Pipe()
    parent_6, child_6 = multiprocessing.Pipe()
    parent_7, child_7 = multiprocessing.Pipe()
    parent_8, child_8 = multiprocessing.Pipe()
    parent_9, child_9 = multiprocessing.Pipe()

    def animation_loop(frame):
        global matrix
        matrix_tmp = np.zeros((n, m))

        p1 = multiprocessing.Process(
            target=matrix_loop_parallel,
            args=(
                0, int(n / 3), 0, int(m / 3), matrix, at, sigma, mu, kernel, count_k, shape_kernel, half_size_i_kernel, half_size_j_kernel, n, m, matrix_tmp, child_1,
            )
        )
        p2 = multiprocessing.Process(
            target=matrix_loop_parallel,
            args=(
                0, int(n / 3), int(m / 3), int(m*2/3), matrix, at, sigma, mu, kernel, count_k, shape_kernel, half_size_i_kernel, half_size_j_kernel, n, m, matrix_tmp, child_2,
            )
        )
        p3 = multiprocessing.Process(
            target=matrix_loop_parallel,
            args=(
                0, int(n / 3), int(m*2/3), m, matrix, at, sigma, mu, kernel, count_k, shape_kernel, half_size_i_kernel, half_size_j_kernel, n, m, matrix_tmp, child_3,
            )
        )
        p4 = multiprocessing.Process(
            target=matrix_loop_parallel,
            args=(
                int(n / 3), int(n*2/3), 0, int(m/3), matrix, at, sigma, mu, kernel, count_k, shape_kernel, half_size_i_kernel, half_size_j_kernel, n, m, matrix_tmp, child_4,
            )
        )
        p5 = multiprocessing.Process(
            target=matrix_loop_parallel,
            args=(
                int(n / 3), int(n*2/3), int(m/3), int(m*2/3), matrix, at, sigma, mu, kernel, count_k, shape_kernel, half_size_i_kernel,
                half_size_j_kernel, n, m, matrix_tmp, child_5,
            )
        )
        p6 = multiprocessing.Process(
            target=matrix_loop_parallel,
            args=(
                int(n / 3), int(n*2/3), int(m*2/3), m, matrix, at, sigma, mu, kernel, count_k, shape_kernel, half_size_i_kernel,
                half_size_j_kernel, n, m, matrix_tmp, child_6,
            )
        )
        p7 = multiprocessing.Process(
            target=matrix_loop_parallel,
            args=(
                int(n*2/3), n, 0, int(m/3), matrix, at, sigma, mu, kernel, count_k, shape_kernel, half_size_i_kernel,
                half_size_j_kernel, n, m, matrix_tmp, child_7,
            )
        )
        p8 = multiprocessing.Process(
            target=matrix_loop_parallel,
            args=(
                int(n*2/3), n, int(m / 3), int(m*2/3), matrix, at, sigma, mu, kernel, count_k, shape_kernel, half_size_i_kernel,
                half_size_j_kernel, n, m, matrix_tmp, child_8,
            )
        )
        p9 = multiprocessing.Process(
            target=matrix_loop_parallel,
            args=(
                int(n*2/3), n, int(m*2/3), m, matrix, at, sigma, mu, kernel, count_k, shape_kernel, half_size_i_kernel,
                half_size_j_kernel, n, m, matrix_tmp, child_9,
            )
        )
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        p5.start()
        p6.start()
        p7.start()
        p8.start()
        p9.start()

        data_1 = parent_1.recv()
        data_2 = parent_2.recv()
        data_3 = parent_3.recv()
        data_4 = parent_4.recv()
        data_5 = parent_5.recv()
        data_6 = parent_6.recv()
        data_7 = parent_7.recv()
        data_8 = parent_8.recv()
        data_9 = parent_9.recv()

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if 0 <= i < int(n / 3) and 0 <= j < int(m / 3):
                    matrix[i][j] = data_1[i][j]
                elif 0 <= i < int(n / 3) and int(m / 3) <= j < int(m*2/3):
                    matrix[i][j] = data_2[i][j]
                elif 0 <= i < int(n / 3) and int(m*2/3) <= j < m:
                    matrix[i][j] = data_3[i][j]
                elif int(n / 3) <= i < int(n*2/3) and 0 <= j < int(m/3):
                    matrix[i][j] = data_4[i][j]
                elif int(n / 3) <= i < int(n*2/3) and int(m/3) <= j < int(m*2/3):
                    matrix[i][j] = data_5[i][j]
                elif int(n / 3) <= i < int(n*2/3) and int(m*2/3) <= j < m:
                    matrix[i][j] = data_6[i][j]
                elif int(n*2/3) <= i < n and 0 <= j < int(m/3):
                    matrix[i][j] = data_7[i][j]
                elif int(n*2/3) <= i < n and int(m / 3) <= j < int(m*2/3):
                    matrix[i][j] = data_8[i][j]
                else:
                    matrix[i][j] = data_9[i][j]

        im.set_array(matrix)
        global count
        plt.title(f'Generation: {count}')
        count += 1
        print(count)
        return im,


    animation = FuncAnimation(fig, func=animation_loop, frames=100, interval=1,
                              cache_frame_data=False)
    # plt.show()
    animation.save('lenia.gif')

