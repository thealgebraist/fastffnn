#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cublas_v2.h>
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
const int MAX_NEURONS = 8192;
const int NUM_CLASSES = 10;
const int TRAIN_LIMIT_S = 600; 
const int INITIAL_NEURONS = 512;
const int MAX_SAMPLES = 50000;
const int BATCH_SIZE = 4096;
const int BLOCK_SIZE = 128;
const int NUM_BLOCKS = 128;

__global__ void gather_images_kernel(const float* all_images, const int* batch_indices, float* batch_imgs, int B, int D) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) {
        int idx = batch_indices[b];
        for(int d=0; d<D; ++d) { batch_imgs[b * D + d] = all_images[idx * D + d]; }
    }
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

__global__ void backprop_intermediate_kernel(const float* dLogits, const float* W2, const float* hs_norm, const float* gamma, const float* var, float* dh_scaled, int H, int B, int C) {
    int b = blockIdx.x;
    int i = threadIdx.x;
    if (b < B && i < H) {
        float dl_dact = 0;
        for(int c=0; c<C; ++c) dl_dact += dLogits[b * C + c] * W2[c * MAX_NEURONS + i];
        float act = hs_norm[b * H + i] * gamma[i];
        float dl_dy = dl_dact * (act > 0 ? 1.0f : 0.1f);
        dh_scaled[b * H + i] = dl_dy / var[i];
    }
}

__global__ void block_newton_kernel(float* W1, const int* indices, const float* batch_imgs, float* dh_scaled, float lr, int H, int B) {
    int block_id = blockIdx.x;
    extern __shared__ float smem[];
    float* H_mat = &smem[0];
    float* g_vec = &smem[128 * 128];
    
    for(int i = threadIdx.x; i < 128; i += blockDim.x) {
        g_vec[i] = 0;
        for(int j=0; j<128; ++j) H_mat[i * 128 + j] = 0;
    }
    __syncthreads();

    for (int b = 0; b < B; ++b) {
        float row_g[128]; 
        for (int k = threadIdx.x; k < 128; k += blockDim.x) {
            int idx = indices[block_id * 128 + k];
            row_g[k] = dh_scaled[b * H + (idx / 3072)] * batch_imgs[b * 3072 + (idx % 3072)];
            atomicAdd(&g_vec[k], row_g[k]);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < 128; i += blockDim.x) {
            for (int j = 0; j < 128; ++j) {
                atomicAdd(&H_mat[i * 128 + j], row_g[i] * row_g[j]); 
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float eps = 1e-3f;
        for(int i=0; i<128; ++i) H_mat[i * 128 + i] += eps;

        for (int i = 0; i < 128; i++) {
            int pivot = i;
            for (int j = i + 1; j < 128; j++) if (fabs(H_mat[j * 128 + i]) > fabs(H_mat[pivot * 128 + i])) pivot = j;
            for (int j = i; j < 128; j++) { 
                float tmp = H_mat[i * 128 + j]; 
                H_mat[i * 128 + j] = H_mat[pivot * 128 + j]; 
                H_mat[pivot * 128 + j] = tmp; 
            }
            float tmp_g = g_vec[i]; g_vec[i] = g_vec[pivot]; g_vec[pivot] = tmp_g;

            for (int j = i + 1; j < 128; j++) {
                float factor = H_mat[j * 128 + i] / H_mat[i * 128 + i];
                g_vec[j] -= factor * g_vec[i];
                for (int k = i; k < 128; k++) H_mat[j * 128 + k] -= factor * H_mat[i * 128 + k];
            }
        }
        for (int i = 127; i >= 0; i--) {
            for (int j = i + 1; j < 128; j++) g_vec[i] -= H_mat[i * 128 + j] * g_vec[j];
            g_vec[i] /= H_mat[i * 128 + i];
        }
        for (int k = 0; k < 128; k++) W1[indices[block_id * 128 + k]] -= lr * g_vec[k];
    }
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
    cout << "Newton Solver (Dynamic SMEM 128-block) H200 Optimized..." << endl;
    download_cifar10();
    CudaVector all_images; vector<uint8_t, CudaManagedAllocator<uint8_t>> all_labels;
    for(int i=1; i<=5; ++i) load_cifar("cifar-10-batches-bin/data_batch_" + to_string(i) + ".bin", all_images, all_labels);
    
    int H = INITIAL_NEURONS;
    CudaVector W1(MAX_NEURONS * INPUT_DIM), b1(MAX_NEURONS, 0), W2(NUM_CLASSES * MAX_NEURONS), b2(NUM_CLASSES, 0), bn_gamma(MAX_NEURONS, 1.0f), bn_beta(MAX_NEURONS, 0.0f);
    mt19937 gen(42); float s1 = sqrtf(2.0f / (INPUT_DIM + H)), s2 = sqrtf(2.0f / (H + NUM_CLASSES)); 
    normal_distribution<float> d1(0, s1), d2(0, s2);
    for(int i=0; i<MAX_NEURONS*INPUT_DIM; ++i) W1[i] = d1(gen);
    for(int i=0; i<NUM_CLASSES*MAX_NEURONS; ++i) W2[i] = d2(gen);

    cublasHandle_t handle; CHECK_CUBLAS(cublasCreate(&handle));
    
    // Set Shared Memory Limit for 128-block (approx 66KB)
    int smem_size = (128 * 128 + 128) * sizeof(float);
    CHECK_CUDA(cudaFuncSetAttribute(block_newton_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    float* loss_gpu; int* correct_gpu; CHECK_CUDA(cudaMallocManaged(&loss_gpu, sizeof(float))); CHECK_CUDA(cudaMallocManaged(&correct_gpu, sizeof(int)));
    int* batch_indices_gpu; CHECK_CUDA(cudaMallocManaged(&batch_indices_gpu, BATCH_SIZE * sizeof(int)));
    int* block_indices_gpu; CHECK_CUDA(cudaMallocManaged(&block_indices_gpu, NUM_BLOCKS * BLOCK_SIZE * sizeof(int)));
    CudaVector batch_imgs(BATCH_SIZE * INPUT_DIM), hs(BATCH_SIZE * MAX_NEURONS), hs_norm(BATCH_SIZE * MAX_NEURONS), logits(BATCH_SIZE * NUM_CLASSES), dLogits(BATCH_SIZE * NUM_CLASSES), dh_scaled(BATCH_SIZE * MAX_NEURONS);
    CudaVector mu(MAX_NEURONS), var(MAX_NEURONS);
    vector<uint8_t, CudaManagedAllocator<uint8_t>> batch_labels(BATCH_SIZE);

    auto start_time = chrono::high_resolution_clock::now();
    int last_s = -1; float lr = 0.5f; int t = 0;

    while (chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start_time).count() < TRAIN_LIMIT_S) {
        *loss_gpu = 0; *correct_gpu = 0;
        for(int b=0; b<BATCH_SIZE; ++b) { batch_indices_gpu[b] = gen() % MAX_SAMPLES; batch_labels[b] = all_labels[batch_indices_gpu[b]]; }
        gather_images_kernel<<<(BATCH_SIZE+255)/256, 256>>>(all_images.data(), batch_indices_gpu, batch_imgs.data(), BATCH_SIZE, INPUT_DIM);

        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, H, BATCH_SIZE, INPUT_DIM, &alpha, W1.data(), MAX_NEURONS, batch_imgs.data(), INPUT_DIM, &beta, hs.data(), H));
        bn_lrelu_forward_kernel<<<(H+255)/256, 256>>>(hs.data(), hs_norm.data(), b1.data(), bn_gamma.data(), bn_beta.data(), mu.data(), var.data(), H, BATCH_SIZE);
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, NUM_CLASSES, BATCH_SIZE, H, &alpha, W2.data(), NUM_CLASSES, hs.data(), H, &beta, logits.data(), NUM_CLASSES));
        softmax_loss_kernel<<<(BATCH_SIZE+255)/256, 256>>>(logits.data(), b2.data(), batch_labels.data(), dLogits.data(), loss_gpu, correct_gpu, BATCH_SIZE, NUM_CLASSES);
        cudaDeviceSynchronize();
        
        backprop_intermediate_kernel<<<BATCH_SIZE, H>>>(dLogits.data(), W2.data(), hs_norm.data(), bn_gamma.data(), var.data(), dh_scaled.data(), H, BATCH_SIZE, NUM_CLASSES);
        for(int i=0; i < NUM_BLOCKS * BLOCK_SIZE; ++i) block_indices_gpu[i] = gen() % (H * INPUT_DIM);
        
        block_newton_kernel<<<NUM_BLOCKS, 256, smem_size>>>(W1.data(), block_indices_gpu, batch_imgs.data(), dh_scaled.data(), lr, H, BATCH_SIZE);
        
        t++; cudaDeviceSynchronize();
        int current_s = chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start_time).count();
        if (current_s > last_s) {
            float err = (1.0f - (*correct_gpu / (float)BATCH_SIZE)) * 100.0f;
            cout << "[Time: " << current_s << "s] Iter: " << t << " | Err: " << err << "% | Block: 128" << endl;
            last_s = current_s;
        }
    }
    cublasDestroy(handle); return 0;
}
