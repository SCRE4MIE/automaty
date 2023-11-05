import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.animation import FuncAnimation
import matplotlib
import multiprocessing
from scipy.ndimage import gaussian_filter
from numba import jit

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


@jit(nopython=True)
def calc_U(matrix, i_c, j_c, kernel, count_k, shape_kernel, half_size_i_kernel, half_size_j_kernel, n, m):
    u = 0
    for i_k in range(shape_kernel[0]):
        for j_k in range(shape_kernel[1]):
            i_matrix_index = i_c - half_size_i_kernel + i_k
            j_matrix_index = j_c - half_size_j_kernel + j_k
            u += matrix[i_matrix_index % n][j_matrix_index % m] * kernel[i_k][j_k]
    u = u / count_k
    return u

@jit(nopython=True)
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

    pipe.send(matrix_tmp[start_i:end_i, start_j:end_j])


if __name__ == '__main__':

    n = 300
    m = 300
    kernel_outer_radius = 12
    kernel_inner_radius = 7
    smoothing_factor = 1
    at = 0.1
    sigma = 0.03
    mu = 0.246

    frames = 10

    kernel = create_kernel(kernel_outer_radius, kernel_inner_radius, smoothing_factor=smoothing_factor)
    matrix = np.zeros((n, m))

    count_k = np.sum(kernel)
    shape_kernel = len(kernel), len(kernel[0])
    half_size_i_kernel = int(shape_kernel[0] / 2)
    half_size_j_kernel = int(shape_kernel[1] / 2)

    #  -------- LOAD POINTS-------
    load_points(matrix=matrix, points_x=[random.randint(0, 299) for _ in range(5000)],
                      points_y=[random.randint(0, 299) for _ in range(5000)])
    load_points(matrix=matrix, points_x=[random.randint(100, 140) for _ in range(1000)],
                points_y=[random.randint(100, 140) for _ in range(1000)])
    load_points(matrix=matrix, points_x=[random.randint(150, 170) for _ in range(1000)],
                points_y=[random.randint(150, 170) for _ in range(1000)])
    load_points(matrix=matrix, points_x=[random.randint(100, 130) for _ in range(2000)],
                points_y=[random.randint(150, 190) for _ in range(2000)])
    # ----------------------------

    fig = plt.figure(figsize=(8, 8))
    im = plt.imshow(matrix, cmap='jet', animated=True)
    plt.axis('off')
    count = 0

    number_of_processes = 9

    # ---- variables to help optimization ---------
    n_3 = int(n/3)
    n_23 = int(n*2/3)
    m_3 = int(m/3)
    m_23 = int(m * 2 / 3)
    # ---------------------------------------------

    # --- PIPE ------------------
    pipes = []
    for p in range(number_of_processes):
        pipes.append(multiprocessing.Pipe())


    def animation_loop(frame):
        global matrix
        matrix_tmp = np.zeros((n, m))

        processes = []
        for p_i in range(number_of_processes):
            processes.append(
                multiprocessing.Process(
                    target=matrix_loop_parallel,
                    args=(
                        0, n, 0, int(m*(p_i + 1)/number_of_processes), matrix, at, sigma, mu, kernel, count_k, shape_kernel, half_size_i_kernel, half_size_j_kernel, n, m, matrix_tmp, pipes[p_i][1],
                    )
                )
            )

        for proces in processes:
            proces.start()

        matrix_tmp_list = []
        for pipe in pipes:
            matrix_tmp_list.append(pipe[0].recv())

        matrix = np.concatenate(matrix_tmp_list, axis=1)
        

        im.set_array(matrix)
        global count
        plt.title(f'Generation: {count}')
        count += 1
        print(count)
        return im,


    animation = FuncAnimation(fig, func=animation_loop, frames=frames, interval=1,
                              cache_frame_data=False, blit=True)

    # plt.show()
    animation.save('lenia.gif')

