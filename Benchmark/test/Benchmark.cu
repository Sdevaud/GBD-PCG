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
// #include <cooperative_groups.h>

// // Distributed Shared memory histogram kernel
// __global__ void clusterHist_kernel(int *bins, const int nbins, const int bins_per_block, const int *__restrict__ input,
//                                    size_t array_size)
// {
//   extern __shared__ int smem[];
//   namespace cg = cooperative_groups;
//   int tid = cg::this_grid().thread_rank();

//   // Cluster initialization, size and calculating local bin offsets.
//   cg::cluster_group cluster = cg::this_cluster();
//   unsigned int clusterBlockRank = cluster.block_rank();
//   int cluster_size = cluster.dim_blocks().x;

//   for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
//   {
//     smem[i] = 0; //Initialize shared memory histogram to zeros
//   }

//   // cluster synchronization ensures that shared memory is initialized to zero in
//   // all thread blocks in the cluster. It also ensures that all thread blocks
//   // have started executing and they exist concurrently.
//   cluster.sync();

//   for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
//   {
//     int ldata = input[i];

//     //Find the right histogram bin.
//     int binid = ldata;
//     if (ldata < 0)
//       binid = 0;
//     else if (ldata >= nbins)
//       binid = nbins - 1;

//     //Find destination block rank and offset for computing
//     //distributed shared memory histogram
//     int dst_block_rank = (int)(binid / bins_per_block);
//     int dst_offset = binid % bins_per_block;

//     //Pointer to target block shared memory
//     int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

//     //Perform atomic update of the histogram bin
//     atomicAdd(dst_smem + dst_offset, 1);
//   }

//   // cluster synchronization is required to ensure all distributed shared
//   // memory operations are completed and no thread block exits while
//   // other thread blocks are still accessing distributed shared memory
//   cluster.sync();

//   // Perform global memory histogram, using the local distributed memory histogram
//   int *lbins = bins + cluster.block_rank() * bins_per_block;
//   for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
//   {
//     atomicAdd(&lbins[i], smem[i]);
//   }
// }

// // Launch via extensible launch
// int main() {
//   cudaLaunchConfig_t config = {0};
//   config.gridDim = array_size / threads_per_block;
//   config.blockDim = threads_per_block;

//   // cluster_size depends on the histogram size.
//   // ( cluster_size == 1 ) implies no distributed shared memory, just thread block local shared memory
//   int cluster_size = 2; // size 2 is an example here
//   int nbins_per_block = nbins / cluster_size;

//   //dynamic shared memory size is per block.
//   //Distributed shared memory size =  cluster_size * nbins_per_block * sizeof(int)
//   config.dynamicSmemBytes = nbins_per_block * sizeof(int);

//   CUDA_CHECK(::cudaFuncSetAttribute((void *)clusterHist_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, config.dynamicSmemBytes));

//   cudaLaunchAttribute attribute[1];
//   attribute[0].id = cudaLaunchAttributeClusterDimension;
//   attribute[0].val.clusterDim.x = cluster_size;
//   attribute[0].val.clusterDim.y = 1;
//   attribute[0].val.clusterDim.z = 1;

//   config.numAttrs = 1;
//   config.attrs = attribute;

//   cudaLaunchKernelEx(&config, clusterHist_kernel, bins, nbins, nbins_per_block, input, array_size);
// }
