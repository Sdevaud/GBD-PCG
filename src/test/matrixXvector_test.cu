// mv_bench.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call) do {                                     \
  cudaError_t err = (call);                                       \
  if (err != cudaSuccess) {                                       \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
            __FILE__, __LINE__, cudaGetErrorString(err));         \
    std::exit(1);                                                 \
  }                                                               \
} while(0)

// ---------------- CPU reference (row-major mat, vec length = 3*b_dim) ----------------
template<typename T>
void cpu_ref_rowmajor(uint32_t b_dim,
                      const T* mat_row,   // [b_dim x 3*b_dim] row-major
                      const T* vec_in,    // [3*b_dim]
                      T* vec_out)         // [b_dim]
{
    const uint32_t cols = 3u * b_dim;
    for (uint32_t r = 0; r < b_dim; r++) {
        T val = (T)0;
        for (uint32_t c = 0; c < cols; c++) {
            val += mat_row[r * cols + c] * vec_in[c];
        }
        vec_out[r] = val;
    }
}

template<typename T>
void check_error(const T* ref, const T* got, uint32_t size, T eps = (T)1e-3)
{
    double err = 0.0;
    for (uint32_t i = 0; i < size; ++i) err += std::abs((double)ref[i] - (double)got[i]);
    if (err > (double)eps) {
        std::cerr << "Mismatch! L1 error = " << err << "\n";
    }
}

template<typename T>
void rowmajor_to_colmajor(const T* mat_row, T* mat_col, uint32_t nbrrow, uint32_t nbrcol)
{
    // mat_row: [nbrrow x nbrcol] row-major
    // mat_col: [nbrrow x nbrcol] column-major => mat_col[r + c*nbrrow] = mat_row[r*nbrcol + c]
    for (uint32_t r = 0; r < nbrrow; ++r) {
        for (uint32_t c = 0; c < nbrcol; ++c) {
            mat_col[r + c * nbrrow] = mat_row[r * nbrcol + c];
        }
    }
}

template<typename T>
T random_normal_clamped(T minv, T maxv)
{
    // Box-Muller (approx)
    const T u1 = (T)rand() / (T)RAND_MAX;
    const T u2 = (T)rand() / (T)RAND_MAX;
    const T pi = (T)3.14159265358979323846;
    T num = std::sqrt((T)-2 * std::log(std::max(u1, (T)1e-12))) * std::cos((T)2 * pi * u2);
    if (num < minv) return minv;
    if (num > maxv) return maxv;
    return num;
}

// ---------------- Device helpers (row-major) ----------------
template<typename T>
__device__ __forceinline__
void mv_rowmajor_simple(T* s_dst,
                        const T* s_mat,   // [b_dim x ld] row-major
                        const T* s_vec,   // [3*b_dim]
                        uint32_t b_dim,
                        uint32_t numCols,
                        uint32_t ld)
{
    for (uint32_t r = threadIdx.x; r < b_dim; r += blockDim.x) {
        T val = (T)0;
        for (uint32_t c = 0; c < numCols; c++) {
            val += s_mat[r * ld + c] * s_vec[c];
        }
        s_dst[r] = val;
    }
}

template<typename T>
__device__ __forceinline__
void mv_rowmajor_shfl(T* s_dst,
                      const T* s_mat,   // [b_dim x (3*b_dim)] row-major, ld = 3*b_dim (no pad)
                      const T* s_vec,
                      uint32_t b_dim)
{
    const uint32_t FULL_MASK = 0xffffffffu;
    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid >> 5;   // /32
    const uint32_t lane    = tid & 31;   // %32
    const uint32_t numCols = 3u * b_dim;
    const uint32_t numWarps = (blockDim.x + 31u) >> 5;

    for (uint32_t r = warp_id; r < b_dim; r += numWarps) {
        T val = (T)0;
        for (uint32_t c = lane; c < numCols; c += 32u) {
            val += s_mat[r * numCols + c] * s_vec[c];
        }
        for (uint32_t off = 16; off > 0; off >>= 1) {
            val += __shfl_down_sync(FULL_MASK, val, off);
        }
        if (lane == 0) s_dst[r] = val;
    }
}

// ---------------- Device helpers (column-major) ----------------
template<typename T>
__device__ __forceinline__
void mv_colmajor_simple(T* s_dst,
                        const T* s_mat_col,  // column-major [b_dim x 3*b_dim] => idx = r + c*b_dim
                        const T* s_vec,
                        uint32_t b_dim)
{
    const uint32_t numCols = 3u * b_dim;
    for (uint32_t r = threadIdx.x; r < b_dim; r += blockDim.x) {
        T val = (T)0;
        for (uint32_t c = 0; c < numCols; c++) {
            val += s_mat_col[r + c * b_dim] * s_vec[c];
        }
        s_dst[r] = val;
    }
}

template<typename T>
__device__ __forceinline__
void mv_colmajor_shfl(T* s_dst,
                      const T* s_mat_col,
                      const T* s_vec,
                      uint32_t b_dim)
{
    const uint32_t FULL_MASK = 0xffffffffu;
    const uint32_t tid     = threadIdx.x;
    const uint32_t warp_id = tid >> 5;
    const uint32_t lane    = tid & 31;
    const uint32_t numCols = 3u * b_dim;
    const uint32_t numWarps = (blockDim.x + 31u) >> 5;

    for (uint32_t r = warp_id; r < b_dim; r += numWarps) {
        T val = (T)0;
        for (uint32_t c = lane; c < numCols; c += 32u) {
            val += s_mat_col[r + c * b_dim] * s_vec[c];
        }
        for (uint32_t off = 16; off > 0; off >>= 1) {
            val += __shfl_down_sync(FULL_MASK, val, off);
        }
        if (lane == 0) s_dst[r] = val;
    }
}

// ---------------- Kernels + cycle counter ----------------
// Convention: cycles_out[0] = elapsed cycles (end-start) for this block (ici grid=1 donc un seul)
template<typename T, uint32_t PAD>
__global__
void kernel_rowmajor_simple(uint32_t b_dim,
                            const T* g_mat_row, // [b_dim x 3*b_dim] row-major
                            const T* g_vec,     // [3*b_dim]
                            T* g_out,           // [b_dim]
                            unsigned long long* cycles_out)
{
    const uint32_t numCols = 3u * b_dim;
    const uint32_t ld = numCols + PAD;

    extern __shared__ unsigned char smem_raw[];
    T* s_dst = reinterpret_cast<T*>(smem_raw);
    T* s_vec = s_dst + b_dim;
    T* s_mat = s_vec + numCols;

    // load vec
    for (uint32_t i = threadIdx.x; i < numCols; i += blockDim.x) {
        s_vec[i] = g_vec[i];
    }

    // load mat with padding into shared
    for (uint32_t idx = threadIdx.x; idx < b_dim * numCols; idx += blockDim.x) {
        uint32_t r = idx / numCols;
        uint32_t c = idx % numCols;
        s_mat[r * ld + c] = g_mat_row[r * numCols + c];
    }
    // clear pad columns
    if constexpr (PAD > 0) {
        for (uint32_t idx = threadIdx.x; idx < b_dim * PAD; idx += blockDim.x) {
            uint32_t r = idx / PAD;
            uint32_t p = idx % PAD;
            s_mat[r * ld + (numCols + p)] = (T)0;
        }
    }

    __syncthreads();

    unsigned long long start = 0, end = 0;
    if (threadIdx.x == 0) start = clock64();
    __syncthreads();

    mv_rowmajor_simple<T>(s_dst, s_mat, s_vec, b_dim, numCols, ld);

    __syncthreads();
    if (threadIdx.x == 0) end = clock64();
    __syncthreads();

    // store out
    for (uint32_t r = threadIdx.x; r < b_dim; r += blockDim.x) g_out[r] = s_dst[r];

    if (threadIdx.x == 0) cycles_out[0] = (end - start);
}

template<typename T>
__global__
void kernel_rowmajor_shfl(uint32_t b_dim,
                          const T* g_mat_row,
                          const T* g_vec,
                          T* g_out,
                          unsigned long long* cycles_out)
{
    const uint32_t numCols = 3u * b_dim;

    extern __shared__ unsigned char smem_raw[];
    T* s_dst = reinterpret_cast<T*>(smem_raw);
    T* s_vec = s_dst + b_dim;
    T* s_mat = s_vec + numCols; // no pad, ld=numCols

    for (uint32_t i = threadIdx.x; i < numCols; i += blockDim.x) s_vec[i] = g_vec[i];
    for (uint32_t idx = threadIdx.x; idx < b_dim * numCols; idx += blockDim.x) s_mat[idx] = g_mat_row[idx];

    __syncthreads();

    unsigned long long start = 0, end = 0;
    if (threadIdx.x == 0) start = clock64();
    __syncthreads();

    mv_rowmajor_shfl<T>(s_dst, s_mat, s_vec, b_dim);

    __syncthreads();
    if (threadIdx.x == 0) end = clock64();
    __syncthreads();

    for (uint32_t r = threadIdx.x; r < b_dim; r += blockDim.x) g_out[r] = s_dst[r];
    if (threadIdx.x == 0) cycles_out[0] = (end - start);
}

template<typename T>
__global__
void kernel_colmajor_simple(uint32_t b_dim,
                            const T* g_mat_col, // column-major [b_dim x 3*b_dim]
                            const T* g_vec,
                            T* g_out,
                            unsigned long long* cycles_out)
{
    const uint32_t numCols = 3u * b_dim;

    extern __shared__ unsigned char smem_raw[];
    T* s_dst = reinterpret_cast<T*>(smem_raw);
    T* s_vec = s_dst + b_dim;
    T* s_mat = s_vec + numCols; // col-major packed

    for (uint32_t i = threadIdx.x; i < numCols; i += blockDim.x) s_vec[i] = g_vec[i];
    for (uint32_t idx = threadIdx.x; idx < b_dim * numCols; idx += blockDim.x) s_mat[idx] = g_mat_col[idx];

    __syncthreads();

    unsigned long long start = 0, end = 0;
    if (threadIdx.x == 0) start = clock64();
    __syncthreads();

    mv_colmajor_simple<T>(s_dst, s_mat, s_vec, b_dim);

    __syncthreads();
    if (threadIdx.x == 0) end = clock64();
    __syncthreads();

    for (uint32_t r = threadIdx.x; r < b_dim; r += blockDim.x) g_out[r] = s_dst[r];
    if (threadIdx.x == 0) cycles_out[0] = (end - start);
}

template<typename T>
__global__
void kernel_colmajor_shfl(uint32_t b_dim,
                          const T* g_mat_col,
                          const T* g_vec,
                          T* g_out,
                          unsigned long long* cycles_out)
{
    const uint32_t numCols = 3u * b_dim;

    extern __shared__ unsigned char smem_raw[];
    T* s_dst = reinterpret_cast<T*>(smem_raw);
    T* s_vec = s_dst + b_dim;
    T* s_mat = s_vec + numCols;

    for (uint32_t i = threadIdx.x; i < numCols; i += blockDim.x) s_vec[i] = g_vec[i];
    for (uint32_t idx = threadIdx.x; idx < b_dim * numCols; idx += blockDim.x) s_mat[idx] = g_mat_col[idx];

    __syncthreads();

    unsigned long long start = 0, end = 0;
    if (threadIdx.x == 0) start = clock64();
    __syncthreads();

    mv_colmajor_shfl<T>(s_dst, s_mat, s_vec, b_dim);

    __syncthreads();
    if (threadIdx.x == 0) end = clock64();
    __syncthreads();

    for (uint32_t r = threadIdx.x; r < b_dim; r += blockDim.x) g_out[r] = s_dst[r];
    if (threadIdx.x == 0) cycles_out[0] = (end - start);
}

// ---------------- Bench ----------------
// result rows: [0]=B_DIM, [1]=shmemKB_row, [2]=cycles_row, [3]=cycles_row_shfl, [4]=cycles_col, [5]=cycles_col_shfl
template<typename T>
std::vector<std::vector<double>> bench(uint32_t threads_per_block, uint32_t nbr_run)
{
    dim3 block(threads_per_block);
    dim3 grid(1);

    std::vector<std::vector<double>> result(6);

    // device buffers reused
    T* d_vec = nullptr;
    T* d_mat_row = nullptr;
    T* d_mat_col = nullptr;
    T* d_out = nullptr;
    unsigned long long* d_cycles = nullptr;

    unsigned long long h_cycles = 0;

    for (uint32_t B_DIM = 5; B_DIM <= 60; B_DIM += 1) {

        const uint32_t cols = 3u * B_DIM;
        const size_t size_out = (size_t)B_DIM * sizeof(T);
        const size_t size_vec = (size_t)cols * sizeof(T);
        const size_t size_mat = (size_t)B_DIM * (size_t)cols * sizeof(T);

        // shared memory sizes (bytes)
        constexpr uint32_t PAD = 1;
        const uint32_t ld_padded = cols + PAD;
        const size_t shBytes_row_simple =
            (size_t)B_DIM * sizeof(T) +                 // dst
            (size_t)cols * sizeof(T) +                  // vec
            (size_t)B_DIM * (size_t)ld_padded * sizeof(T); // mat padded

        const size_t shBytes_row_shfl =
            (size_t)B_DIM * sizeof(T) +
            (size_t)cols * sizeof(T) +
            (size_t)B_DIM * (size_t)cols * sizeof(T);

        const size_t shBytes_col =
            (size_t)B_DIM * sizeof(T) +
            (size_t)cols * sizeof(T) +
            (size_t)B_DIM * (size_t)cols * sizeof(T);

        double acc_row = 0.0, acc_row_shfl = 0.0, acc_col = 0.0, acc_col_shfl = 0.0;

        // host buffers per B_DIM
        std::vector<T> h_vec(cols), h_out(B_DIM), h_ref(B_DIM);
        std::vector<T> h_mat_row(size_mat / sizeof(T));
        std::vector<T> h_mat_col(size_mat / sizeof(T));

        // init random
        for (uint32_t c = 0; c < cols; ++c) h_vec[c] = random_normal_clamped<T>((T)-100, (T)100);
        for (uint32_t i = 0; i < (uint32_t)(h_mat_row.size()); ++i) h_mat_row[i] = random_normal_clamped<T>((T)-100, (T)100);

        cpu_ref_rowmajor<T>(B_DIM, h_mat_row.data(), h_vec.data(), h_ref.data());
        rowmajor_to_colmajor<T>(h_mat_row.data(), h_mat_col.data(), B_DIM, cols);

        // alloc device (per B_DIM, simplest)
        CUDA_CHECK(cudaMalloc(&d_vec, size_vec));
        CUDA_CHECK(cudaMalloc(&d_mat_row, size_mat));
        CUDA_CHECK(cudaMalloc(&d_mat_col, size_mat));
        CUDA_CHECK(cudaMalloc(&d_out, size_out));
        CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));

        CUDA_CHECK(cudaMemcpy(d_vec, h_vec.data(), size_vec, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mat_row, h_mat_row.data(), size_mat, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mat_col, h_mat_col.data(), size_mat, cudaMemcpyHostToDevice));

        // --- crée les events une fois (par B_DIM c'est ok) ---
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        float total_ms = 0.0f;

        // ---------------- Row-major simple ----------------
        kernel_rowmajor_simple<T, PAD><<<grid, block, shBytes_row_simple>>>(B_DIM, d_mat_row, d_vec, d_out, d_cycles);
        CUDA_CHECK(cudaDeviceSynchronize()); // warmup

        CUDA_CHECK(cudaEventRecord(start));
        for (uint32_t run = 0; run < nbr_run; ++run) {
            kernel_rowmajor_simple<T, PAD><<<grid, block, shBytes_row_simple>>>(B_DIM, d_mat_row, d_vec, d_out, d_cycles);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        double avg_row_ms = (double)total_ms / (double)nbr_run;

        // (optionnel) check erreur 1 seule fois (pas dans la mesure)
        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, size_out, cudaMemcpyDeviceToHost));
        check_error<T>(h_ref.data(), h_out.data(), B_DIM);

        // ---------------- Row-major shfl ----------------
        kernel_rowmajor_shfl<T><<<grid, block, shBytes_row_shfl>>>(B_DIM, d_mat_row, d_vec, d_out, d_cycles);
        CUDA_CHECK(cudaDeviceSynchronize()); // warmup

        CUDA_CHECK(cudaEventRecord(start));
        for (uint32_t run = 0; run < nbr_run; ++run) {
            kernel_rowmajor_shfl<T><<<grid, block, shBytes_row_shfl>>>(B_DIM, d_mat_row, d_vec, d_out, d_cycles);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        double avg_row_shfl_ms = (double)total_ms / (double)nbr_run;

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, size_out, cudaMemcpyDeviceToHost));
        check_error<T>(h_ref.data(), h_out.data(), B_DIM);

        // ---------------- Col-major simple ----------------
        kernel_colmajor_simple<T><<<grid, block, shBytes_col>>>(B_DIM, d_mat_col, d_vec, d_out, d_cycles);
        CUDA_CHECK(cudaDeviceSynchronize()); // warmup

        CUDA_CHECK(cudaEventRecord(start));
        for (uint32_t run = 0; run < nbr_run; ++run) {
            kernel_colmajor_simple<T><<<grid, block, shBytes_col>>>(B_DIM, d_mat_col, d_vec, d_out, d_cycles);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        double avg_col_ms = (double)total_ms / (double)nbr_run;

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, size_out, cudaMemcpyDeviceToHost));
        check_error<T>(h_ref.data(), h_out.data(), B_DIM);

        // ---------------- Col-major shfl ----------------
        kernel_colmajor_shfl<T><<<grid, block, shBytes_col>>>(B_DIM, d_mat_col, d_vec, d_out, d_cycles);
        CUDA_CHECK(cudaDeviceSynchronize()); // warmup

        CUDA_CHECK(cudaEventRecord(start));
        for (uint32_t run = 0; run < nbr_run; ++run) {
            kernel_colmajor_shfl<T><<<grid, block, shBytes_col>>>(B_DIM, d_mat_col, d_vec, d_out, d_cycles);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        double avg_col_shfl_ms = (double)total_ms / (double)nbr_run;

        CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, size_out, cudaMemcpyDeviceToHost));
        check_error<T>(h_ref.data(), h_out.data(), B_DIM);

        // --- détruit events ---
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // stocke dans result (remplace cycles par ms moyen)
        result[0].push_back((double)B_DIM);
        result[1].push_back((double)shBytes_row_simple / 1024.0);
        result[2].push_back(avg_row_ms);
        result[3].push_back(avg_row_shfl_ms);
        result[4].push_back(avg_col_ms);
        result[5].push_back(avg_col_shfl_ms);

        printf("B_DIM=%u | shRowSimple=%.2f KB | Row=%.6f ms | RowShfl=%.6f ms | Col=%.6f ms | ColShfl=%.6f ms\n",
            B_DIM,
            (double)shBytes_row_simple / 1024.0,
            avg_row_ms,
            avg_row_shfl_ms,
            avg_col_ms,
            avg_col_shfl_ms);

        // free per B_DIM
        CUDA_CHECK(cudaFree(d_vec)); d_vec = nullptr;
        CUDA_CHECK(cudaFree(d_mat_row)); d_mat_row = nullptr;
        CUDA_CHECK(cudaFree(d_mat_col)); d_mat_col = nullptr;
        CUDA_CHECK(cudaFree(d_out)); d_out = nullptr;
        CUDA_CHECK(cudaFree(d_cycles)); d_cycles = nullptr;
    }

    return result;
}

template<typename T>
void write_result_txt(const std::vector<std::vector<T>>& result, const char* filename = "results.txt")
{
    FILE* f = std::fopen(filename, "w");
    if (!f) { std::perror("fopen"); return; }

    std::fprintf(f, "# Columns: B_DIM, shmemKB_rowSimple, cycles_row, cycles_row_shfl, cycles_col, cycles_col_shfl\n");
    const size_t n = result[0].size();
    for (size_t i = 0; i < n; ++i) {
        std::fprintf(f, "%.0f %.9f %.9f %.9f %.9f %.9f\n",
                     (double)result[0][i],
                     (double)result[1][i],
                     (double)result[2][i],
                     (double)result[3][i],
                     (double)result[4][i],
                     (double)result[5][i]);
    }
    std::fclose(f);
}

int main()
{
    std::srand(0);
    auto result = bench<float>(1024, 1000);

    // write out
    write_result_txt<double>(result, "results.txt");

    return 0;
}
