const canvas = document.getElementById("border");
const canvasKernel = document.getElementById("kernelType");
const ctx = canvas.getContext("2d");
const ctx_kernel = canvasKernel.getContext("2d");

const n = 300;
const m = 300;


let rules_born = [3]
let rules_die = [2,3]


// Function to create an initial matrix
const createMatrixZeros = (rows, cols) => {
    let matrix = new Array(rows);
    for (let i = 0; i < rows; i++) {
        matrix[i] = new Array(cols).fill(0); // Initialize with zeros
    }
    return matrix;
}

let matrix = createMatrixZeros(n, m);


const loadPoints = (array_x, array_y, matrix) => {
    for (let i = 0; i < array_x.length; i++) {
        matrix[array_y[i]][array_x[i]] = 1
    }
    return matrix
}

matrix = loadPoints(
    [1, 2, 1, 2, 11, 11, 11, 12, 12, 13, 13, 14, 14, 15, 16, 16, 17, 17, 17, 18, 21, 21, 21, 22, 22, 22, 23, 23, 25, 25, 25, 25, 35, 35, 36, 36, 100, 100, 101, 100, 99],
    [30, 30, 31, 31, 30, 31, 32, 29, 33, 28, 34, 28, 34, 31, 29, 33, 30, 32, 31, 31, 30, 29, 28, 30, 29, 28, 27, 31, 31, 32, 27, 26, 28, 29, 28, 29, 100, 99, 99, 101, 100],
    matrix
)



const drawMatrix = (ctx, canvas,  matrix, n, m) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let cellSize = canvas.width / m;

    for (let row = 0; row < n; row++) {
        for (let col = 0; col < m; col++) {
            let valueCell = matrix[row][col];
            let channel = parseInt(255 * valueCell);
            ctx.fillStyle = `rgb(${channel},${channel},${channel})`;


            ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
        }
    }
}



const create_kernel = (outer_radius, inner_radius) => {
    let size = (2 * outer_radius) - 1
    let kernel = createMatrixZeros(size, size);
    let center = outer_radius - 1

    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            let distance = Math.sqrt((i - center) ** 2 + (j - center) ** 2)
            if (distance < inner_radius) {
                kernel[i][j] = 0
            } else if (distance < outer_radius) {
                kernel[i][j] = 1
            }
        }
    }

    return kernel
}

let kernel = create_kernel(2, 1);

const count_cells = (matrix, i_c, j_c, kernel, n, m) => {
    let count = 0
    let shape_kernel = [kernel.length, kernel[0].length]
    let half_size_i_kernel = parseInt(shape_kernel[0] / 2)
    let half_size_j_kernel = parseInt(shape_kernel[1] / 2)

    for (let i_k = 0; i_k < shape_kernel[0]; i_k++) {
        for (let j_k = 0; j_k < shape_kernel[1]; j_k++) {
            let i_matrix_index = i_c - half_size_i_kernel + i_k
            let j_matrix_index = j_c - half_size_j_kernel + j_k
            count += matrix[(i_matrix_index % n + n) % n][(j_matrix_index % m + m) % m] * kernel[i_k][j_k]
        }
    }
    return count
}

const check_born_or_die = (i, j, matrix, kernel, n, m, rules_born, rules_die) => {
    let count = count_cells(matrix, i, j, kernel, n, m)
    if (matrix[i][j] === 0) {

        if (rules_born.includes(count)) {
            return 1
        } else {
            return 0
        }
    }

    if (matrix[i][j] === 1) {

        if (!(rules_die.includes(count))) {
            return 0
        } else {
            return 1
        }
    }
}

const gameOfLifeLoop = (matrix, kernel, n, m, rules_born, rules_die) => {
    let matrix_tmp = createMatrixZeros(n, m)
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[0].length; j++) {
            matrix_tmp[i][j] = check_born_or_die(i, j, matrix, kernel, n, m, rules_born, rules_die)
        }
    }
    return matrix_tmp.slice(0)
}

const mainLoop = () => {
    matrix = gameOfLifeLoop(matrix, kernel, n, m, rules_born, rules_die);
    drawMatrix(ctx, canvas, matrix, n, m);
    requestAnimationFrame(mainLoop);
}

mainLoop();
let kernel_2 = create_kernel(12, 5);
drawMatrix(ctx_kernel, canvasKernel, kernel, kernel.length, kernel[0].length);
