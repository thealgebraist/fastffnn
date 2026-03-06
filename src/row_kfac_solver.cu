#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <sys/stat.h>
#include <cstdlib>

using namespace std;

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(1); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if(status != CUBLAS_STATUS_SUCCESS) { \
        cerr << "CUBLAS error at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(1); \
    } \
}

#define CHECK_CUSOLVER(call) { \
    cusolverStatus_t status = call; \
    if(status != CUSOLVER_STATUS_SUCCESS) { \
        cerr << "CUSOLVER error at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(1); \
    } \
}

template <class T>
struct CudaManagedAllocator {
    typedef T value_type;
    CudaManagedAllocator() = default;
    template <class U> constexpr CudaManagedAllocator(const CudaManagedAllocator<U>&) noexcept {}
    T* allocate(std::size_t n) {
        T* p;
        CHECK_CUDA(cudaMallocManaged(&p, n * sizeof(T)));
        return p;
    }
    void deallocate(T* p, std::size_t n) noexcept { cudaFree(p); }
};
using CudaVector = std::vector<float, CudaManagedAllocator<float>>;

const int INPUT_DIM = 3072;
const int MAX_NEURONS = 1024;
const int NUM_CLASSES = 10;
const int TRAIN_LIMIT_S = 600; 
const int BATCH_SIZE = 50000;

__global__ void add_eye_kernel(float* A, float lambda, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i * n + i] += lambda;
}

__global__ void bn_lrelu_forward_kernel(float* hs, float* hs_norm, const float* b1, const float* gamma, const float* beta, float* mu, float* var, int H, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H) {
        float m = 0, v = 0;
        for(int b=0; b<B; ++b) { float val = hs[b * H + i] + b1[i]; hs[b * H + i] = val; m += val; }
        m /= B; mu[i] = m;
        for(int b=0; b<B; ++b) { float d = hs[b * H + i] - m; v += d * d; }
        v = sqrtf(v / B + 1e-5f); var[i] = v;
        for(int b=0; b<B; ++b) {
            hs_norm[b * H + i] = (hs[b * H + i] - m) / v;
            float act = hs_norm[b * H + i] * gamma[i] + beta[i];
            hs[b * H + i] = act > 0 ? act : 0.1f * act;
        }
    }
}

__global__ void softmax_loss_kernel(const float* logits, const float* b2, const uint8_t* labels, float* dLogits, float* loss, int* correct, int B, int C) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) {
        float max_l = -1e10;
        for(int c=0; c<C; ++c) { float l = logits[b * C + c] + b2[c]; if(l > max_l) max_l = l; }
        float sum_e = 0;
        for(int c=0; c<C; ++c) sum_e += expf((logits[b * C + c] + b2[c]) - max_l);
        int label = labels[b];
        float prob = expf((logits[b * C + label] + b2[label]) - max_l) / sum_e;
        atomicAdd(loss, -logf(prob + 1e-9f));
        int pred = 0; float max_p = -1.0;
        for(int c=0; c<C; ++c) {
            float p = expf((logits[b * C + c] + b2[c]) - max_l) / sum_e;
            dLogits[b * C + c] = p - (c == label ? 1.0f : 0.0f);
            if(p > max_p) { max_p = p; pred = c; }
        }
        if(pred == label) atomicAdd(correct, 1);
    }
}

__global__ void get_neuron_grad_kernel(const float* dLogits, const float* W2, const float* hs_norm, const float* gamma, const float* var, float* da_i, int target_row, int H, int B, int C) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) {
        float dl_dact = 0;
        for(int c=0; c<C; ++c) dl_dact += dLogits[b * C + c] * W2[c * H + target_row];
        float act = hs_norm[b * H + target_row] * gamma[target_row];
        da_i[b] = (dl_dact * (act > 0 ? 1.0f : 0.1f)) / var[target_row];
    }
}

__global__ void update_row_kernel(float* W1_row, const float* dw_i, float lr, int target_row, int D) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < D) {
        W1_row[target_row * D + j] -= lr * dw_i[j];
    }
}

void invert_matrix_cusolver(cusolverDnHandle_t handle, float* A, float* A_inv, int n, float lambda) {
    add_eye_kernel<<<(n+255)/256, 256>>>(A, lambda, n);
    int work_size = 0;
    CHECK_CUSOLVER(cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, n, A, n, &work_size));
    float* work; cudaMalloc(&work, work_size * sizeof(float));
    int* info; cudaMallocManaged(&info, sizeof(int));
    cudaMemset(A_inv, 0, n * n * sizeof(float));
    for(int i=0; i<n; ++i) { float one = 1.0f; cudaMemcpy(A_inv + i*n + i, &one, sizeof(float), cudaMemcpyHostToDevice); }
    CHECK_CUSOLVER(cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_LOWER, n, A, n, work, work_size, info));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUSOLVER(cusolverDnSpotrs(handle, CUBLAS_FILL_MODE_LOWER, n, n, A, n, A_inv, n, info));
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(work); cudaFree(info);
}

void download_cifar10() {
    struct stat buffer;
    if (stat("cifar-10-batches-bin/data_batch_1.bin", &buffer) != 0) {
        system("wget -qO- https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz | tar xz");
    }
}

bool load_cifar(const string& path, CudaVector& images, vector<uint8_t, CudaManagedAllocator<uint8_t>>& labels) {
    ifstream file(path, ios::binary); if (!file) return false;
    for (int i = 0; i < 10000; ++i) {
        uint8_t label; file.read((char*)&label, 1); labels.push_back(label);
        vector<uint8_t> raw(3072); file.read((char*)raw.data(), 3072);
        for(int p=0; p<3072; ++p) images.push_back(raw[p] / 255.0f);
    }
    return true;
}

int main() {
    cout << "Single-Neuron Row K-FAC Solver (H200)..." << endl;
    download_cifar10();
    CudaVector all_images; vector<uint8_t, CudaManagedAllocator<uint8_t>> all_labels;
    for(int i=1; i<=5; ++i) load_cifar("cifar-10-batches-bin/data_batch_" + to_string(i) + ".bin", all_images, all_labels);
    
    int H = MAX_NEURONS;
    CudaVector W1(H * INPUT_DIM), b1(H, 0), W2(NUM_CLASSES * H), b2(NUM_CLASSES, 0), bn_gamma(H, 1.0f), bn_beta(H, 0.0f);
    mt19937 gen(42); float s1 = sqrtf(2.0f / (INPUT_DIM + H)), s2 = sqrtf(2.0f / (H + NUM_CLASSES)); 
    normal_distribution<float> d1(0, s1), d2(0, s2);
    for(int i=0; i<H*INPUT_DIM; ++i) W1[i] = d1(gen);
    for(int i=0; i<NUM_CLASSES*H; ++i) W2[i] = d2(gen);

    cublasHandle_t handle; CHECK_CUBLAS(cublasCreate(&handle));
    cusolverDnHandle_t solver_handle; CHECK_CUSOLVER(cusolverDnCreate(&solver_handle));
    float* loss_gpu; int* correct_gpu; CHECK_CUDA(cudaMallocManaged(&loss_gpu, sizeof(float))); CHECK_CUDA(cudaMallocManaged(&correct_gpu, sizeof(int)));
    
    CudaVector hs(BATCH_SIZE * H), hs_norm(BATCH_SIZE * H), logits(BATCH_SIZE * NUM_CLASSES), dLogits(BATCH_SIZE * NUM_CLASSES);
    CudaVector mu(H), var(H), da_i(BATCH_SIZE), row_grad(INPUT_DIM), row_natural_grad(INPUT_DIM);
    CudaVector A(INPUT_DIM * INPUT_DIM), A_inv(INPUT_DIM * INPUT_DIM);

    // Initial A inversion (entire dataset covariance)
    cout << "Computing and inverting global input covariance A..." << endl;
    float alpha = 1.0f / BATCH_SIZE, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, INPUT_DIM, INPUT_DIM, BATCH_SIZE, &alpha, all_images.data(), INPUT_DIM, all_images.data(), INPUT_DIM, &beta, A.data(), INPUT_DIM));
    invert_matrix_cusolver(solver_handle, A.data(), A_inv.data(), INPUT_DIM, 0.01f);

    auto start_time = chrono::high_resolution_clock::now();
    int last_s = -1; float lr = 0.1f; int t = 0;

    while (chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start_time).count() < TRAIN_LIMIT_S) {
        *loss_gpu = 0; *correct_gpu = 0;
        
        // Forward
        float f_alpha = 1.0f, f_beta = 0.0f;
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, H, BATCH_SIZE, INPUT_DIM, &f_alpha, W1.data(), H, all_images.data(), INPUT_DIM, &f_beta, hs.data(), H));
        bn_lrelu_forward_kernel<<<(H+255)/256, 256>>>(hs.data(), hs_norm.data(), b1.data(), bn_gamma.data(), bn_beta.data(), mu.data(), var.data(), H, BATCH_SIZE);
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, NUM_CLASSES, BATCH_SIZE, H, &f_alpha, W2.data(), NUM_CLASSES, hs.data(), H, &f_beta, logits.data(), NUM_CLASSES));
        softmax_loss_kernel<<<(BATCH_SIZE+255)/256, 256>>>(logits.data(), b2.data(), all_labels.data(), dLogits.data(), loss_gpu, correct_gpu, BATCH_SIZE, NUM_CLASSES);
        cudaDeviceSynchronize();

        // 1. Pick a random neuron row
        int target_row = gen() % H;

        // 2. Get local pre-activation gradients for that neuron
        get_neuron_grad_kernel<<<(BATCH_SIZE+255)/256, 256>>>(dLogits.data(), W2.data(), hs_norm.data(), bn_gamma.data(), var.data(), da_i.data(), target_row, H, BATCH_SIZE, NUM_CLASSES);
        
        // 3. Compute row gradient: g_i = mean(da_i * X)
        CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, INPUT_DIM, BATCH_SIZE, &alpha, all_images.data(), INPUT_DIM, da_i.data(), 1, &f_beta, row_grad.data(), 1));
        
        // 4. Compute Fisher scalar s_i = mean(da_i^2)
        float s_i;
        CHECK_CUBLAS(cublasSnrm2(handle, BATCH_SIZE, da_i.data(), 1, &s_i));
        s_i = (s_i * s_i) / BATCH_SIZE + 1e-6f;

        // 5. Precondition: dw_i = row_grad * A_inv / s_i
        float inv_si = 1.0f / s_i;
        CHECK_CUBLAS(cublasSgemv(handle, CUBLAS_OP_N, INPUT_DIM, INPUT_DIM, &inv_si, A_inv.data(), INPUT_DIM, row_grad.data(), 1, &f_beta, row_natural_grad.data(), 1));

        // 6. Update targeted row
        update_row_kernel<<<(INPUT_DIM+255)/256, 256>>>(W1.data(), row_natural_grad.data(), lr, target_row, INPUT_DIM);

        t++; cudaDeviceSynchronize();
        int current_s = chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start_time).count();
        if (current_s > last_s) {
            float err = (1.0f - (*correct_gpu / (float)BATCH_SIZE)) * 100.0f;
            cout << "[Time: " << current_s << "s] Iter: " << t << " | Err: " << err << "% | Row: " << target_row << " | s_i: " << s_i << endl;
            last_s = current_s;
        }
    }
    cublasDestroy(handle); cusolverDnDestroy(solver_handle); return 0;
}
