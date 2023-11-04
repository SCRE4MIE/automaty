import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import multiprocessing
import random
import time

from matplotlib.animation import FuncAnimation

matplotlib.use('macosx')
# matplotlib.use('TkAgg')
# matplotlib.use("Agg")


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


def translate_rules(rules_str):
    split_rules_str = rules_str.split('/')
    rules_to_die = [int(c) for c in split_rules_str[0]]
    rules_to_born = [int(c) for c in split_rules_str[1]]
    return rules_to_born, rules_to_die


def create_kernel(outer_radius, inner_radius):
    size = 2 * outer_radius - 1
    kernel = np.zeros((size, size))
    center = outer_radius - 1

    for i in range(size):
        for j in range(size):
            distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            if distance < inner_radius:
                kernel[i][j] = 0
            elif distance < outer_radius:
                kernel[i][j] = 1

    return kernel


def count_cells(matrix, i_c, j_c, kernel, n, m):
    count = 0
    shape_kernel = len(kernel), len(kernel[0])
    half_size_i_kernel = int(shape_kernel[0] / 2)
    half_size_j_kernel = int(shape_kernel[1] / 2)

    for i_k in range(shape_kernel[0]):
        for j_k in range(shape_kernel[1]):
            i_matrix_index = i_c - half_size_i_kernel + i_k
            j_matrix_index = j_c - half_size_j_kernel + j_k
            count += matrix[i_matrix_index % n][j_matrix_index % m] * kernel[i_k][j_k]

    return count


def check_born_or_die(i, j, matrix, kernel, n, m, rules_born, rules_die):
    count = count_cells(matrix=matrix, i_c=i, j_c=j, kernel=kernel, n=n, m=m)

    if matrix[i][j] == 0:  # born
        if count in rules_born:
            return 1
        else:
            return 0

    if matrix[i][j] == 1:  # die
        if count not in rules_die:
            return 0
        else:
            return 1


def task(start_i, end_i, start_j, end_j, matrix, kernel, n, m, rules_born, rules_die, matrix_tmp, pipe):
    for i_o in range(start_i, end_i):
        for j_o in range(start_j, end_j):
            matrix_tmp[i_o][j_o] = check_born_or_die(i=i_o, j=j_o, matrix=matrix, kernel=kernel, n=n, m=m,
                                                     rules_born=rules_born, rules_die=rules_die)
    pipe.send(matrix_tmp)


if __name__ == '__main__':
    rules_born, rules_die = translate_rules('23/3')
    n = 600
    m = 600
    kernel = create_kernel(outer_radius=10, inner_radius=1)
    matrix = np.zeros((n, m))
    # load_file(matrix, 'data.dat')
    load_points(matrix, points_x=[random.randint(0, 599) for _ in range(1000)],
                points_y=[random.randint(0, 599) for _ in range(1000)])

    conn1_1, conn2_1 = multiprocessing.Pipe()
    conn1_2, conn2_2 = multiprocessing.Pipe()
    conn1_3, conn2_3 = multiprocessing.Pipe()
    conn1_4, conn2_4 = multiprocessing.Pipe()

    fig = plt.figure(figsize=(8, 8))
    im = plt.imshow(matrix, cmap='jet', animated=True)
    plt.axis('off')
    count = 0


    def animation_loop(frame):
        start = time.process_time()
        global matrix
        matrix_tmp = np.zeros((n, m))
        p1 = multiprocessing.Process(target=task, args=(
        0, int(n / 2), 0, int(m / 2), matrix, kernel, n, m, rules_born, rules_die, matrix_tmp, conn2_1))
        p2 = multiprocessing.Process(target=task, args=(
        0, int(n / 2), int(m / 2), m, matrix, kernel, n, m, rules_born, rules_die, matrix_tmp, conn2_2))
        p3 = multiprocessing.Process(target=task, args=(
        int(n / 2), n, 0, int(m / 2), matrix, kernel, n, m, rules_born, rules_die, matrix_tmp, conn2_3))
        p4 = multiprocessing.Process(target=task, args=(
        int(n / 2), n, int(m / 2), m, matrix, kernel, n, m, rules_born, rules_die, matrix_tmp, conn2_4))
        p1.start()
        p2.start()
        p3.start()
        p4.start()

        data = conn1_1.recv()
        data_2 = conn1_2.recv()
        data_3 = conn1_3.recv()
        data_4 = conn1_4.recv()

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if 0 <= i < int(n / 2) and 0 <= j < int(m / 2):
                    matrix[i][j] = data[i][j]
                elif 0 <= i < int(n / 2) and int(m / 2) <= j < m:
                    matrix[i][j] = data_2[i][j]
                elif int(n / 2) <= i < n and 0 <= j < int(m / 2):
                    matrix[i][j] = data_3[i][j]
                else:
                    matrix[i][j] = data_4[i][j]

        im.set_array(matrix)
        global count
        plt.title(f'Generation: {count}')
        count += 1
        print(time.process_time() - start)
        print(count)

        return im,


    animation = FuncAnimation(fig, func=animation_loop, frames=30, interval=1,
                              cache_frame_data=False)
    plt.show()
    # animation.save('gof.gif')
