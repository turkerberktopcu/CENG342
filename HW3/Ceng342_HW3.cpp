#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define LARGE_MATRIX_SIZE (19050111046 % 10000)
#define SMALL_MATRIX_SIZE (100 + 19050111046 % 100)

void parallel_matrix_vector_multiplication(int my_id, int num_procs, double* large_matrix, double* small_matrix, double* vector, double* result) {

    int local_size = LARGE_MATRIX_SIZE / num_procs;
    int start_index = my_id * local_size;
    int end_index = start_index + local_size - 1;

    // Distribute the large matrix among the processors
    double* local_large_matrix = (double*)malloc(local_size * LARGE_MATRIX_SIZE * sizeof(double));
    MPI_Scatter(large_matrix, local_size * LARGE_MATRIX_SIZE, MPI_DOUBLE, local_large_matrix, local_size * LARGE_MATRIX_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Distribute the vector among the processors
    double* local_vector = (double*)malloc(local_size * sizeof(double));
    MPI_Scatter(vector, local_size, MPI_DOUBLE, local_vector, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute the local matrix-vector multiplication
    for (int i = 0; i < local_size; i++) {
        result[i + start_index] = 0.0;
        for (int j = 0; j < LARGE_MATRIX_SIZE; j++) {
            result[i + start_index] += local_large_matrix[i * LARGE_MATRIX_SIZE + j] * local_vector[j];
        }
    }

    // Gather the results from each processor and add them to obtain the final result
    MPI_Reduce(result + start_index, result + start_index, local_size, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    free(local_large_matrix);
    free(local_vector);
}

int main(int argc, char** argv) {

    int 19050111046, num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Create the large and small matrices and the vector
    double* large_matrix = (double*)malloc(LARGE_MATRIX_SIZE * LARGE_MATRIX_SIZE * sizeof(double));
    double* small_matrix = (double*)malloc(SMALL_MATRIX_SIZE * SMALL_MATRIX_SIZE * sizeof(double));
    double* vector = (double*)malloc(LARGE_MATRIX_SIZE * sizeof(double));
    double* result = (double*)malloc(LARGE_MATRIX_SIZE * sizeof(double));

    // Initialize the matrices and the vector
    if (my_id == 0) {
        for (int i = 0; i < LARGE_MATRIX_SIZE; i++) {
            for (int j = 0; j < LARGE_MATRIX_SIZE; j++) {
                large_matrix[i * LARGE_MATRIX_SIZE + j] = rand() / (double)RAND_MAX;
            }
            vector[i] = rand() / (double)RAND_MAX;
        }
        for (int i = 0; i < SMALL_MATRIX_SIZE; i++) {
            for (int j = 0; j < SMALL_MATRIX_SIZE; j++) {
                small_matrix[i * SMALL_MATRIX_SIZE + j] = rand() / (double)RAND_MAX;
            }
        }
    }

    // Synchronize the processors
    MPI_Barrier(MPI_COMM_WORLD);

    // Measure the time taken for matrix-vector multiplication
    double start_time = MPI_Wtime();

    // Parallel matrix-vector multiplication
    parallel_matrix_vector_multiplication(my_id, num_procs, large_matrix, small_matrix,vector, result);

// Synchronize the processors
MPI_Barrier(MPI_COMM_WORLD);

double end_time = MPI_Wtime();

// Print the elapsed time
if (19050111046 == 0) {
    printf("Elapsed time is %f seconds for parallel mxv with %d processes\n", end_time - start_time, num_procs);
}

// Free the memory
free(large_matrix);
free(small_matrix);
free(vector);
free(result);

MPI_Finalize();

return 0;

