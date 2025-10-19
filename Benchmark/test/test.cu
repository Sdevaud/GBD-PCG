#include <cuda.h>
#include <cooperative_groups.h>
#include <iostream>
#include "test.cuh"


namespace cg = cooperative_groups;


// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}
// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}
// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];
    return Asub;
}

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
    cudaMemcpyHostToDevice);
    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);
    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}
// Matrix multiplication kernel called by MatMul()
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}


// Kernel déclaré avec tailles de bloc et cluster
__block_size__((BLOCK_SIZE, BLOCK_SIZE, 1), (CLUSTER_DIM, CLUSTER_DIM, 1))
__global__ void matmul_cluster_kernel(const float* A, const float* B, float* C, int N) {
    // Groupe de threads du cluster (permet sync inter-blocs)
    cg::cluster_group cluster = cg::this_cluster();

    // Coordonnées du bloc dans la grille de clusters
    int clusterX = cluster.block_rank() % CLUSTER_DIM;
    int clusterY = cluster.block_rank() / CLUSTER_DIM;


    // Coordonnées du cluster dans la grille totale
    int globalClusterX = blockIdx.x / CLUSTER_DIM;
    int globalClusterY = blockIdx.y / CLUSTER_DIM;

    // Coordonnées locales dans le bloc
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    

    // Taille totale du cluster (en éléments)
    int clusterTileSize = BLOCK_SIZE * CLUSTER_DIM;

    // Coordonnées globales de ce thread dans C
    int row = globalClusterY * clusterTileSize + clusterY * BLOCK_SIZE + ty;
    int col = globalClusterX * clusterTileSize + clusterX * BLOCK_SIZE + tx;

    // DSM (Distributed Shared Memory) – partagée entre blocs du cluster
    __shared__ __attribute__((annotate("cluster_shared"))) float tileA[CLUSTER_DIM * BLOCK_SIZE][BLOCK_SIZE];
    __shared__ __attribute__((annotate("cluster_shared"))) float tileB[BLOCK_SIZE][CLUSTER_DIM * BLOCK_SIZE];

    float sum = 0.0f;
    
    // Boucle sur les sous-blocs à multiplier
    for (int m = 0; m < N / (BLOCK_SIZE * CLUSTER_DIM); ++m) { // correction sur les clusters
        // Chaque bloc charge une portion distincte de A et B
        int aRow = row;
        int aCol = m * BLOCK_SIZE + tx;
        int bRow = m * BLOCK_SIZE + ty;
        int bCol = col;

        // On remplit la DSM (partagée à l’échelle du cluster)
        tileA[clusterY * BLOCK_SIZE + ty][tx] = A[aRow * N + aCol];
        tileB[ty][clusterX * BLOCK_SIZE + tx] = B[bRow * N + bCol];

        // Synchronisation inter-blocs (garantie matérielle sur Hopper)
        cluster.sync();

        // Multiplication sur le tile complet partagé
        for (int k = 0; k < BLOCK_SIZE * CLUSTER_DIM; ++k) {
            sum += tileA[(clusterY * BLOCK_SIZE + ty)][k] *
                   tileB[k][(clusterX * BLOCK_SIZE + tx)];
        }

        cluster.sync(); // avant de réécrire la DSM
    }

    // Écriture du résultat
    C[row * N + col] = sum;
}

void create_matrix(float* mat, int N) {
    for (int i = 0; i < N * N; i++) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX; // valeurs aléatoires entre 0 et 1
    }
}

void matmul(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0f;
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

void check_matrix(const float* A, const float* B, int N) {
  float tol = 1e-3;
  bool correct = true;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (fabs(A[i*N + j] - B[i*N + j]) > tol) {
        correct = false;
        std::cout << "Mismatch at (" << i << ", " << j << "): " 
                  << A[i*N + j] << " != " << B[i*N + j] << std::endl;
      }
    }
  }

  if (correct) {
    std::cout << "Matrix multiplication is correct!" << std::endl;
  } else {
    std::cout << "Matrix multiplication is incorrect!" << std::endl;
  }
  return;
      
}

void test_matmul() {
    const int N = 1024;  // Exemple plus petit pour test (peut aller à 16k si VRAM OK)
    int size = N * N * sizeof(float);

    float* A = new float[N * N];
    float* B = new float[N * N];
    create_matrix(A, N);
    create_matrix(B, N);

    // test GPU without cluster
    float *C = new float[N * N];
    for (int i = 0; i < N * N; i++) C[i] = 0.0f;
    float *dA, *dB, *dC;

    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);
    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C, size, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(N / block.x, N / block.y);
    MatMulKernel<<<grid, block>>>(Matrix{N, N, N, dA},
                                  Matrix{N, N, N, dB},
                                  Matrix{N, N, N, dC});
    cudaDeviceSynchronize();
    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); 

    // test GPU with cluster
    float* C_2 = new float[N * N];
    for (int i = 0; i < N * N; i++) C_2[i] = 0.0f;
    float* dC_2;

    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC_2, size);
    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dC_2, C_2, size, cudaMemcpyHostToDevice);

    // Dimensions bloc/grille
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);      // 16×16 threads
    dim3 cluster(CLUSTER_DIM, CLUSTER_DIM, 1);  // 2×2 blocs/cluster
    dim3 grid(N / (BLOCK_SIZE * CLUSTER_DIM),
              N / (BLOCK_SIZE * CLUSTER_DIM), 1); // nombre total de clusters

    // Lancement (avec CUDA 12+, __block_size__ gère la config)
    matmul_cluster_kernel<<<grid, block>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();
    cudaMemcpy(C_2, dC_2, size, cudaMemcpyDeviceToHost);



    // Vérification
    float* C_ref = new float[N * N];
    for (int i = 0; i < N * N; i++) C_ref[i] = 0.0f;
    matmul(A, B, C_ref, N);
    check_matrix((float*)C_ref, (float*)C, N);
    check_matrix((float*)C, (float*)C_2, N);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] A; delete[] B; delete[] C; delete[] C_2; delete[] C_ref;
    return;
}

__global__ void kernel_max_shared(float *out) {
    extern __shared__ float sdata[];

    int idx = threadIdx.x;
    sdata[idx % (blockDim.x)] = (float)idx;

    // Exemple de calcul fictif
    out[blockIdx.x * blockDim.x + idx] = sdata[idx % blockDim.x] * 2.0f;
}

int main() {

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int numSM = prop.multiProcessorCount;
  int sizeSM = prop.sharedMemPerMultiprocessor;
  int numthreadsPerSM = prop.maxThreadsPerMultiProcessor;

  dim3 blockDim(20);
  dim3 gridDim(numSM);

  float *d_out;
  cudaMalloc(&d_out, sizeof(float) * blockDim.x * gridDim.x);

  cudaFuncSetAttribute(kernel_max_shared,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      sizeSM);

  
  int activeBlocksPerSM = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &activeBlocksPerSM, kernel_max_shared, blockDim.x, sizeSM);

  printf("Blocs actifs par SM (occupancy) : %d\n", activeBlocksPerSM);
  printf("Nombre de SM: %d\n", numSM);
  printf("Taille de la mémoire partagée par SM: %d\n", sizeSM);
  printf("numthreadsPerSM: %d\n", numthreadsPerSM);
  printf("Lancement du kernel avec %d blocs et %d threads/bloc...\n",
           gridDim.x, blockDim.x);

  cudaDeviceSynchronize();

  cudaFree(d_out);

  return 0;
}