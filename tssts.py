import numpy as np

# Create a list of 2D numpy arrays
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6]])
matrix2 = np.vstack([matrix2, [0, 0]])  # Add a row of zeros to match the number of rows in matrix1
matrix3 = np.array([[7], [8]])

matrix_list = [matrix1, matrix2, matrix3]

# Use numpy.concatenate to concatenate them horizontally
merged_matrix = np.concatenate(matrix_list, axis=1)

print(merged_matrix)