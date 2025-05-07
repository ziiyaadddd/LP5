cuda_code = """
#include <stdio.h>
#include <cuda.h>

__global__ void vectorAdd(int *a, int *b, int *c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int N = 10;
    size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * (N-i);
    }

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    // Copy result back
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    printf("Vector A: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_a[i]);
    }
    printf("\\n");

    printf("Vector B: ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_b[i]);
    }
    printf("\\n");

    printf("Resultant Vector C (A + B): ");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_c[i]);
    }

    // Cleanup
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
"""

# Write the code to a .cu file
with open('vector_add.cu', 'w') as f:
    f.write(cuda_code)

    # Compile the CUDA code
!nvcc addition.cu -o addition

# Run the compiled code
!./addition