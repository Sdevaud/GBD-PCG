#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

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






// #include <iostream>
// #include <cooperative_groups.h>
// using namespace cooperative_groups;


// __global__ __cluster_dims__(2, 1, 1)
// void clusterSharedDemo() {
//     extern __shared__ float cluster_smem[];
//     cluster_group cluster = this_cluster();
//     int total = cluster.dim_blocks().x * blockDim.x;
//     int global_idx = cluster.block_rank() * blockDim.x + threadIdx.x;

//     cluster_smem[global_idx] = (float)global_idx;
//     cluster.sync();

//     if (cluster.block_rank() == 0 && threadIdx.x == 0) {
//         printf("Cluster shared memory active, total threads: %d\n", total);
//     }
// }

// int main() {
//     int dev = 0;
//     cudaSetDevice(dev);

//     int maxClusterSmem = 0;
//     cudaDeviceGetAttribute(&maxClusterSmem,
//         cudaDevAttrMaxSharedMemoryPerClusterOptin, dev);

//     std::cout << "Max shared memory per cluster: "
//               << maxClusterSmem / 1024 << " KB\n";

//     // On demande le max disponible
//     cudaFuncSetAttribute(clusterSharedDemo,
//         cudaFuncAttributeMaxDynamicSharedMemorySize,
//         maxClusterSmem);

//     int threads = 256;
//     int blocks = 2; // 2 blocs par cluster (défini par __cluster_dims__)

//     void *args[] = {};
//     cudaLaunchKernel((void*)clusterSharedDemo,
//                      dim3(blocks), dim3(threads),
//                      args, maxClusterSmem, 0);

//     cudaDeviceSynchronize();
// }
























// #include <iostream>
// #include <stdio.h>
// #include "gpu_pcg.cuh"
// #include "gpuassert.cuh"
// #include "Gauss_Jordan.cuh"
// #include "affichage.cuh"
// #include "conjugate_gradient.cuh"
// #include "test.cuh"


// int main() {
//   //testing_CG_no_GPU();
//   //test_2x2();
//   // test_pcg();
//   test_matmul();
  

//     return 0;
// }


