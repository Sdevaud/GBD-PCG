#include <cstdio>
#include <cuda.h>

int main() {
    int device;
    cudaGetDevice(&device);

    int clusterLaunch = 0;
    int distributedSmem = 0;
    int maxClusterSize = 0;

    cudaDeviceGetAttribute(&clusterLaunch, cudaDevAttrClusterLaunch, device);
    cudaDeviceGetAttribute(&distributedSmem, cudaDevAttrDistributedSharedMemorySupported, device);
    cudaDeviceGetAttribute(&maxClusterSize, cudaDevAttrMaxClustersPerMultiprocessor, device);

    printf("=== GPU Cluster / DSM Capabilities ===\n");
    printf("Cluster Launch Supported : %s\n", clusterLaunch ? "YES" : "NO");
    printf("Distributed Shared Memory (DSM) : %s\n", distributedSmem ? "YES" : "NO");
    printf("Max Clusters per SM : %d\n", maxClusterSize);

    if(clusterLaunch && distributedSmem)
        printf("\n✅ Ton GPU supporte les clusters + DSM !\n");
    else
        printf("\n❌ Ton GPU ne supporte pas DSM/Clusters (besoin d’un GPU Hopper SM90+)\n");

    return 0;
}
