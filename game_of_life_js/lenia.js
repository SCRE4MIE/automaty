const canvas = document.getElementById("border");
const canvasKernel = document.getElementById("kernelType");
const ctx = canvas.getContext("2d");
const ctx_kernel = canvasKernel.getContext("2d");

const n = 300;
const m = 300;
let kernel_inner_radius = 10;
let kernel_outer_radius = 13;
let smooth = 4;
let aT = 0.1;
let sigma = 0.014;
let mu = 0.15;



// ------------
function gaussianFilter(matrix, sigma) {
  const width = matrix[0].length;
  const height = matrix.length;
  const size = Math.ceil(6 * sigma);

  function gaussian(x, y, sigma) {
    const exponent = -((x ** 2 + y ** 2) / (2 * sigma ** 2));
    return (1 / (2 * Math.PI * sigma ** 2)) * Math.exp(exponent);
  }

  const kernel = new Array(size);

  // Generate the 2D Gaussian kernel
  for (let x = -size; x <= size; x++) {
    kernel[x] = new Array(size);
    for (let y = -size; y <= size; y++) {
      kernel[x][y] = gaussian(x, y, sigma);
    }
  }

  // Normalize the kernel
  const kernelSum = kernel.flat().reduce((acc, val) => acc + val, 0);
  for (let x = -size; x <= size; x++) {
    for (let y = -size; y <= size; y++) {
      kernel[x][y] /= kernelSum;
    }
  }

  const result = new Array(height);

  // Apply the filter to the input matrix
  for (let i = 0; i < height; i++) {
    result[i] = new Array(width).fill(0);
    for (let j = 0; j < width; j++) {
      for (let x = -size; x <= size; x++) {
        for (let y = -size; y <= size; y++) {
          if (i + x >= 0 && i + x < height && j + y >= 0 && j + y < width) {
            result[i][j] += matrix[i + x][j + y] * kernel[x][y];
          }
        }
      }
    }
  }

  return result;
}

// ---------------------



let startAnim = false;

document.getElementById('startBtn').addEventListener("click", (e) =>{
    startAnim = true;
    requestAnimationFrame(mainLoop);
})

document.getElementById('stopBtn').addEventListener("click", (e)=>{
    startAnim = false;
})

document.getElementById('restartBtn').addEventListener("click", (e)=>{
    startAnim = false;
    matrix = createMatrixZeros(n, m);
    matrix = fillMatrix(matrix, 40, 140, 40, 140);
    drawMatrix(ctx, canvas, matrix, n, m);

})



formValues = document.getElementById("formValues");
formValues.addEventListener("submit", (e) => {
    e.preventDefault();
    const dataForm = new FormData(e.target);
    kernel_inner_radius = dataForm.get('kernel_inner_radius');
    kernel_outer_radius = dataForm.get('kernel_outer_radius');
    smooth = dataForm.get('kernel_smooth');
    aT = dataForm.get('deltaT');
    sigma = dataForm.get('sigma');
    mu = dataForm.get('mu');

    kernel = create_kernel(kernel_inner_radius, kernel_outer_radius);
    count_k = sumMatrixValues(kernel);

    shape_kernel = [kernel.length, kernel[0].length];
    half_size_i_kernel = parseInt(shape_kernel[0] / 2);
    half_size_j_kernel = parseInt(shape_kernel[1] / 2);
    drawMatrix(ctx_kernel, canvasKernel, kernel, kernel.length, kernel[0].length);
})


// Function to create an initial matrix
const createMatrixZeros = (rows, cols) => {
    let matrix = new Array(rows);
    for (let i = 0; i < rows; i++) {
        matrix[i] = new Array(cols).fill(0); // Initialize with zeros
    }
    return matrix;
}


const fillMatrix = (matrix, start_n, stop_n, start_m, stop_m) =>{

  for (let i = start_n; i < stop_n; i++) {
    for (let j = start_m; j < stop_m; j++) {
      matrix[i][j] = Math.random(); // Generate a random number between 0 and 1
    }

  }

  return matrix;
}

let matrix = createMatrixZeros(n, m);
matrix = fillMatrix(matrix, 40, 140, 40, 140);


const loadPoints = (array_x, array_y, matrix) => {
    for (let i = 0; i < array_x.length; i++) {
        matrix[array_y[i]][array_x[i]] = 1
    }
    return matrix
}

// matrix = loadPoints(
//     [1, 2, 1, 2, 11, 11, 11, 12, 12, 13, 13, 14, 14, 15, 16, 16, 17, 17, 17, 18, 21, 21, 21, 22, 22, 22, 23, 23, 25, 25, 25, 25, 35, 35, 36, 36, 100, 100, 101, 100, 99],
//     [30, 30, 31, 31, 30, 31, 32, 29, 33, 28, 34, 28, 34, 31, 29, 33, 30, 32, 31, 31, 30, 29, 28, 30, 29, 28, 27, 31, 31, 32, 27, 26, 28, 29, 28, 29, 100, 99, 99, 101, 100],
//     matrix
// )



const drawMatrix = (ctx, canvas,  matrix, n, m) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    let cellSize = canvas.width / m;

    for (let row = 0; row < n; row++) {
        for (let col = 0; col < m; col++) {
            let valueCell = matrix[row][col];
            let channel = parseInt(255 * valueCell);
            ctx.fillStyle = `hsl(${240 - channel}, 100%, 50%)`;


            ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
        }
    }
}



const create_kernel = (inner_radius, outer_radius) => {
    let size = (2 * outer_radius) + 7
    let kernel = createMatrixZeros(size, size);
    let center = outer_radius - 1 + 4

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

    return gaussianFilter(kernel, smooth);
}

const sumMatrixValues  = a => a.reduce((r, x) => r + x.reduce((s, y) => s + y, 0), 0);


let kernel = create_kernel(kernel_inner_radius, kernel_outer_radius);
let count_k = sumMatrixValues(kernel);

let shape_kernel = [kernel.length, kernel[0].length]
let half_size_i_kernel = parseInt(shape_kernel[0] / 2)
let half_size_j_kernel = parseInt(shape_kernel[1] / 2)

const calcU = (matrix, i_c, j_c, kernel, count_k, shape_kernel, half_size_i_kernel, half_size_j_kernel, n, m) => {
    let u = 0

    for (let i_k = 0; i_k < shape_kernel[0]; i_k++) {
        for (let j_k = 0; j_k < shape_kernel[1]; j_k++) {
            let i_matrix_index = i_c - half_size_i_kernel + i_k
            let j_matrix_index = j_c - half_size_j_kernel + j_k
            u += matrix[(i_matrix_index % n + n) % n][(j_matrix_index % m + m) % m] * kernel[i_k][j_k]
        }
    }
    u = u / count_k
    return u
}

const growthFunction = (u, sigma, mu) => {
    let l = Math.abs(u - mu)
    let k = 2 * (sigma ** 2)
    return 2 * Math.exp(-(l ** 2) / k) - 1
}

const calcCT = (matrix, i, j, aT, sigma, mu, kernel, count_k, shape_kernel, half_size_i_kernel, half_size_j_kernel, n, m) => {
    let u = calcU(matrix, i, j, kernel, count_k, shape_kernel, half_size_i_kernel, half_size_j_kernel, n, m);
    let a = growthFunction(u, sigma, mu);
    return Math.min(Math.max((matrix[i][j] + aT * a), 0), 1)
}

const leniaLoop = (matrix, aT, sigma, mu, kernel, count_k, shape_kernel, half_size_i_kernel, half_size_j_kernel, n, m) => {
    let matrix_tmp = createMatrixZeros(n, m)
    for (let i = 0; i < matrix.length; i++) {
        for (let j = 0; j < matrix[0].length; j++) {
            matrix_tmp[i][j] = calcCT(matrix, i, j, aT, sigma, mu, kernel, count_k, shape_kernel, half_size_i_kernel, half_size_j_kernel, n, m)
        }
    }
    return matrix_tmp.slice(0)
}


const mainLoop = () => {
    drawMatrix(ctx, canvas, matrix, n, m);
    if(startAnim === true) {
        matrix = leniaLoop(matrix, aT, sigma, mu, kernel, count_k, shape_kernel, half_size_i_kernel, half_size_j_kernel, n, m);
        requestAnimationFrame(mainLoop);
    }
}

mainLoop();
drawMatrix(ctx_kernel, canvasKernel, kernel, kernel.length, kernel[0].length);
