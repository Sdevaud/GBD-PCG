#include <iostream>
#include <cooperative_groups.h>
using namespace cooperative_groups;

__global__ __cluster_dims__(2, 1, 1)
void clusterSharedDemo() {
    extern __shared__ float cluster_smem[];
    cluster_group cluster = this_cluster();
    int total = cluster.dim_blocks().x * blockDim.x;
    int global_idx = cluster.block_rank() * blockDim.x + threadIdx.x;

    cluster_smem[global_idx] = (float)global_idx;
    cluster.sync();

    if (cluster.block_rank() == 0 && threadIdx.x == 0) {
        printf("Cluster shared memory active, total threads: %d\n", total);
    }
}

int main() {
    int dev = 0;
    cudaSetDevice(dev);

    int maxClusterSmem = 0;
    cudaDeviceGetAttribute(&maxClusterSmem,
        cudaDevAttrMaxSharedMemoryPerClusterOptin, dev);

    std::cout << "Max shared memory per cluster: "
              << maxClusterSmem / 1024 << " KB\n";

    // On demande le max disponible
    cudaFuncSetAttribute(clusterSharedDemo,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        maxClusterSmem);

    int threads = 256;
    int blocks = 2; // 2 blocs par cluster (défini par __cluster_dims__)

    void *args[] = {};
    cudaLaunchKernel((void*)clusterSharedDemo,
                     dim3(blocks), dim3(threads),
                     args, maxClusterSmem, 0);

    cudaDeviceSynchronize();
}
























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


