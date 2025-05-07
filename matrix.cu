# Define the CUDA code for matrix multiplication
cuda_code = """
#include <stdio.h>
#include <cuda.h>

#define N 3 // Matrix size N x N

__global__ void matrixMul(int *A, int *B, int *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    if (row < width && col < width) {
        for (int k = 0; k < width; ++k) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

int main() {
    int size = N * N * sizeof(int);

    // Host memory allocation
    int *h_A = (int*)malloc(size);
    int *h_B = (int*)malloc(size);
    int *h_C = (int*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 10-i;
        h_B[i] = 14-i;
    }

    // Device memory allocation
    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print matrices
    printf("Matrix A:\\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", h_A[i * N + j]);
        }
        printf("\\n");
    }
    printf("\\n");

    printf("Matrix B:\\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", h_B[i * N + j]);
        }
        printf("\\n");
    }
    printf("\\n");

    // Print full matrix
    printf("Resultant Matrix C (A x B):\\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%d ", h_C[i * N + j]);
        }
        printf("\\n");
    }

    // Cleanup
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
"""

# Write the CUDA code to a .cu file
with open('matrix_mul.cu', 'w') as f:
    f.write(cuda_code)

# Compile the CUDA code
!nvcc matrix_mul.cu -o matrix_mul

# Run the compiled code
!./matrix_mul
