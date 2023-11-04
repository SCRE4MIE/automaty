import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import multiprocessing
import random
import time

from matplotlib.animation import FuncAnimation

# matplotlib.use('macosx')
matplotlib.use('TkAgg')
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


def matrix_loop_parallel(start_i, end_i, start_j, end_j, matrix, kernel, n, m, rules_born, rules_die, matrix_tmp, pipe):
    for i_o in range(start_i, end_i):
        for j_o in range(start_j, end_j):
            matrix_tmp[i_o][j_o] = check_born_or_die(i=i_o, j=j_o, matrix=matrix, kernel=kernel, n=n, m=m,
                                                     rules_born=rules_born, rules_die=rules_die)
    pipe.send(matrix_tmp)


if __name__ == '__main__':
    rules_born, rules_die = translate_rules('23/3')
    n = 200
    m = 200
    kernel = create_kernel(outer_radius=2, inner_radius=1)
    matrix = np.zeros((n, m))
    load_file(matrix, 'data.dat')
    # load_points(matrix, points_x=[random.randint(0, 599) for _ in range(1000)],
    #             points_y=[random.randint(0, 599) for _ in range(1000)])

    parent_1, child_1 = multiprocessing.Pipe()
    parent_2, child_2 = multiprocessing.Pipe()
    parent_3, child_3 = multiprocessing.Pipe()
    parent_4, child_4 = multiprocessing.Pipe()
    parent_5, child_5 = multiprocessing.Pipe()
    parent_6, child_6 = multiprocessing.Pipe()
    parent_7, child_7 = multiprocessing.Pipe()
    parent_8, child_8 = multiprocessing.Pipe()
    parent_9, child_9 = multiprocessing.Pipe()

    fig = plt.figure(figsize=(8, 8))
    im = plt.imshow(matrix, cmap='jet', animated=True)
    plt.axis('off')
    count = 0


    def animation_loop(frame):
        start = time.process_time()
        global matrix
        matrix_tmp = np.zeros((n, m))

        p1 = multiprocessing.Process(
            target=matrix_loop_parallel,
            args=(
                0, int(n / 3), 0, int(m / 3), matrix, kernel, n, m, rules_born, rules_die, matrix_tmp, child_1,
            )
        )
        p2 = multiprocessing.Process(
            target=matrix_loop_parallel,
            args=(
                0, int(n / 3), int(m / 3), int(m * 2 / 3), matrix, kernel, n, m, rules_born, rules_die, matrix_tmp, child_2,
            )
        )
        p3 = multiprocessing.Process(
            target=matrix_loop_parallel,
            args=(
                0, int(n / 3), int(m * 2 / 3), m, matrix, kernel, n, m, rules_born, rules_die, matrix_tmp, child_3,
            )
        )
        p4 = multiprocessing.Process(
            target=matrix_loop_parallel,
            args=(
                int(n / 3), int(n * 2 / 3), 0, int(m / 3), matrix, kernel, n, m, rules_born, rules_die, matrix_tmp, child_4,
            )
        )
        p5 = multiprocessing.Process(
            target=matrix_loop_parallel,
            args=(
                int(n / 3), int(n * 2 / 3), int(m / 3), int(m * 2 / 3), matrix, kernel, n, m, rules_born, rules_die, matrix_tmp, child_5,
            )
        )
        p6 = multiprocessing.Process(
            target=matrix_loop_parallel,
            args=(
                int(n / 3), int(n * 2 / 3), int(m * 2 / 3), m, matrix, kernel, n, m, rules_born, rules_die, matrix_tmp, child_6,
            )
        )
        p7 = multiprocessing.Process(
            target=matrix_loop_parallel,
            args=(
                int(n * 2 / 3), n, 0, int(m / 3), matrix, kernel, n, m, rules_born, rules_die, matrix_tmp, child_7,
            )
        )
        p8 = multiprocessing.Process(
            target=matrix_loop_parallel,
            args=(
                int(n * 2 / 3), n, int(m / 3), int(m * 2 / 3), matrix, kernel, n, m, rules_born, rules_die, matrix_tmp, child_8,
            )
        )
        p9 = multiprocessing.Process(
            target=matrix_loop_parallel,
            args=(
                int(n * 2 / 3), n, int(m * 2 / 3), m, matrix, kernel, n, m, rules_born, rules_die, matrix_tmp, child_9,
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
                elif 0 <= i < int(n / 3) and int(m / 3) <= j < int(m * 2 / 3):
                    matrix[i][j] = data_2[i][j]
                elif 0 <= i < int(n / 3) and int(m * 2 / 3) <= j < m:
                    matrix[i][j] = data_3[i][j]
                elif int(n / 3) <= i < int(n * 2 / 3) and 0 <= j < int(m / 3):
                    matrix[i][j] = data_4[i][j]
                elif int(n / 3) <= i < int(n * 2 / 3) and int(m / 3) <= j < int(m * 2 / 3):
                    matrix[i][j] = data_5[i][j]
                elif int(n / 3) <= i < int(n * 2 / 3) and int(m * 2 / 3) <= j < m:
                    matrix[i][j] = data_6[i][j]
                elif int(n * 2 / 3) <= i < n and 0 <= j < int(m / 3):
                    matrix[i][j] = data_7[i][j]
                elif int(n * 2 / 3) <= i < n and int(m / 3) <= j < int(m * 2 / 3):
                    matrix[i][j] = data_8[i][j]
                else:
                    matrix[i][j] = data_9[i][j]

        im.set_array(matrix)
        global count
        plt.title(f'Generation: {count}')
        count += 1
        print(time.process_time() - start)
        print(count)

        return im,


    animation = FuncAnimation(fig, func=animation_loop, frames=200, interval=1,
                              cache_frame_data=False)
    # plt.show()
    animation.save('gof.gif')
