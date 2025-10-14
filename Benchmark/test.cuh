#define BLOCK_SIZE 16
#define CLUSTER_DIM 2 

void test_matmul();

//__block_size__((BLOCK_SIZE, BLOCK_SIZE, 1), (CLUSTER_DIM, CLUSTER_DIM, 1))
__global__ void matmul_cluster_kernel(const float* A, const float* B, float* C, int N);



typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

void check_matrix(const float* A, const float* B, int N) ;
void create_matrix(float* mat, int N);
void matmul(const float* A, const float* B, float* C, int N);
void matmul_cluster(const float* A, const float* B, float* C, int N);