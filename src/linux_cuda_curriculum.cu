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

__global__ void block_newton_kernel(float* W1, const int* indices, const float* batch_imgs, float* dh_scaled, float* global_H, float* global_g, float lr, int H, int B, int cur_b_size, bool use_global) {
    int block_id = blockIdx.x;
    extern __shared__ float smem[];
    float* H_mat; float* g_vec;
    if (use_global) {
        H_mat = &global_H[block_id * cur_b_size * cur_b_size];
        g_vec = &global_g[block_id * cur_b_size];
    } else {
        H_mat = &smem[0];
        g_vec = &smem[cur_b_size * cur_b_size];
    }
    
    if(threadIdx.x == 0) {
        for(int i=0; i<cur_b_size; ++i) {
            g_vec[i] = 0;
            for(int j=0; j<cur_b_size; ++j) H_mat[i * cur_b_size + j] = 0;
        }
    }
    __syncthreads();

    for (int b = threadIdx.x; b < B; b += blockDim.x) {
        float sample_g[512]; // Stack limited, but small enough for registers if cur_b_size is small. For 512, will spill.
        // Optimization: direct accumulation
        for (int k = 0; k < cur_b_size; ++k) {
            int idx = indices[block_id * cur_b_size + k];
            int row = idx / 3072; int col = idx % 3072;
            float gk = dh_scaled[b * H + row] * batch_imgs[b * 3072 + col];
            atomicAdd(&g_vec[k], gk);
            for (int j = 0; j < cur_b_size; ++j) {
                // Approximate Hessian via Outer Product (Natural Gradient)
                float gj = dh_scaled[b * H + (indices[block_id * cur_b_size + j] / 3072)] * batch_imgs[b * 3072 + (indices[block_id * cur_b_size + j] % 3072)];
                atomicAdd(&H_mat[k * cur_b_size + j], gk * gj);
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float eps = 1e-3f;
        for(int i=0; i<cur_b_size; ++i) H_mat[i * cur_b_size + i] += eps;

        for (int i = 0; i < cur_b_size; i++) {
            int pivot = i;
            for (int j = i + 1; j < cur_b_size; j++) 
                if (fabs(H_mat[j * cur_b_size + i]) > fabs(H_mat[pivot * cur_b_size + i])) pivot = j;
            
            for (int j = i; j < cur_b_size; j++) {
                float tmp = H_mat[i * cur_b_size + j];
                H_mat[i * cur_b_size + j] = H_mat[pivot * cur_b_size + j];
                H_mat[pivot * cur_b_size + j] = tmp;
            }
            float tmp_g = g_vec[i]; g_vec[i] = g_vec[pivot]; g_vec[pivot] = tmp_g;

            for (int j = i + 1; j < cur_b_size; j++) {
                float factor = H_mat[j * cur_b_size + i] / H_mat[i * cur_b_size + i];
                g_vec[j] -= factor * g_vec[i];
                for (int k = i; k < cur_b_size; k++) H_mat[j * cur_b_size + k] -= factor * H_mat[i * cur_b_size + k];
            }
        }
        for (int i = cur_b_size - 1; i >= 0; i--) {
            for (int j = i + 1; j < cur_b_size; j++) g_vec[i] -= H_mat[i * cur_b_size + j] * g_vec[j];
            g_vec[i] /= H_mat[i * cur_b_size + i];
        }
        for (int k = 0; k < cur_b_size; k++) {
            int idx = indices[block_id * cur_b_size + k];
            W1[idx] -= lr * g_vec[k];
        }
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
    cout << "Comprehensive Newton Block-Size Benchmark (2 to 512 vars, 20s total)..." << endl;
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
    float* loss_gpu; int* correct_gpu; CHECK_CUDA(cudaMallocManaged(&loss_gpu, sizeof(float))); CHECK_CUDA(cudaMallocManaged(&correct_gpu, sizeof(int)));
    int* batch_indices_gpu; CHECK_CUDA(cudaMallocManaged(&batch_indices_gpu, BATCH_SIZE * sizeof(int)));
    int* block_indices_gpu; CHECK_CUDA(cudaMallocManaged(&block_indices_gpu, 16384 * sizeof(int)));
    float* global_H; CHECK_CUDA(cudaMalloc(&global_H, 16384 * 512 * sizeof(float)));
    float* global_g; CHECK_CUDA(cudaMalloc(&global_g, 16384 * sizeof(float)));
    
    CudaVector batch_imgs(BATCH_SIZE * INPUT_DIM), hs(BATCH_SIZE * MAX_NEURONS), hs_norm(BATCH_SIZE * MAX_NEURONS), logits(BATCH_SIZE * NUM_CLASSES), dLogits(BATCH_SIZE * NUM_CLASSES), dh_scaled(BATCH_SIZE * MAX_NEURONS);
    CudaVector mu(MAX_NEURONS), var(MAX_NEURONS);
    vector<uint8_t, CudaManagedAllocator<uint8_t>> batch_labels(BATCH_SIZE);

    vector<int> b_configs = {2, 4, 8, 16, 32, 64, 128, 256, 512};
    int best_b = 16; float best_eff = -1.0f;
    auto total_bench_start = chrono::high_resolution_clock::now();

    for (int b_size : b_configs) {
        if (chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - total_bench_start).count() > 20) break;
        auto b_start = chrono::high_resolution_clock::now();
        float acc_start = 0;
        int iters = (b_size > 128) ? 2 : 5;
        for (int i=0; i < iters; ++i) {
            *loss_gpu = 0; *correct_gpu = 0;
            for(int b=0; b<BATCH_SIZE; ++b) { batch_indices_gpu[b] = gen() % MAX_SAMPLES; batch_labels[b] = all_labels[batch_indices_gpu[b]]; }
            gather_images_kernel<<<(BATCH_SIZE+255)/256, 256>>>(all_images.data(), batch_indices_gpu, batch_imgs.data(), BATCH_SIZE, INPUT_DIM);
            float alpha = 1.0f, beta = 0.0f;
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, H, BATCH_SIZE, INPUT_DIM, &alpha, W1.data(), MAX_NEURONS, batch_imgs.data(), INPUT_DIM, &beta, hs.data(), H));
            bn_lrelu_forward_kernel<<<(H+255)/256, 256>>>(hs.data(), hs_norm.data(), b1.data(), bn_gamma.data(), bn_beta.data(), mu.data(), var.data(), H, BATCH_SIZE);
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, NUM_CLASSES, BATCH_SIZE, H, &alpha, W2.data(), NUM_CLASSES, hs.data(), H, &beta, logits.data(), NUM_CLASSES));
            softmax_loss_kernel<<<(BATCH_SIZE+255)/256, 256>>>(logits.data(), b2.data(), batch_labels.data(), dLogits.data(), loss_gpu, correct_gpu, BATCH_SIZE, NUM_CLASSES);
            cudaDeviceSynchronize();
            if (i==0) acc_start = (*correct_gpu / (float)BATCH_SIZE);
            backprop_intermediate_kernel<<<BATCH_SIZE, H>>>(dLogits.data(), W2.data(), hs_norm.data(), bn_gamma.data(), var.data(), dh_scaled.data(), H, BATCH_SIZE, NUM_CLASSES);
            int n_blocks = 16384 / b_size;
            for(int k=0; k < n_blocks * b_size; ++k) block_indices_gpu[k] = gen() % (H * INPUT_DIM);
            bool use_global = (b_size > 128);
            int smem_size = use_global ? 0 : (b_size * b_size + b_size) * sizeof(float);
            block_newton_kernel<<<n_blocks, 256, smem_size>>>(W1.data(), block_indices_gpu, batch_imgs.data(), dh_scaled.data(), global_H, global_g, 0.5f, H, BATCH_SIZE, b_size, use_global);
            cudaDeviceSynchronize();
        }
        float acc_end = (*correct_gpu / (float)BATCH_SIZE);
        double elapsed = chrono::duration<double>(chrono::high_resolution_clock::now() - b_start).count();
        float eff = (acc_end - acc_start) / elapsed;
        cout << "[Bench] Block Size: " << b_size << " | Efficiency: " << eff << " ΔAcc/s" << endl;
        if (eff > best_eff) { best_eff = eff; best_b = b_size; }
    }
    cout << "Winner: Block Size " << best_b << endl;

    auto start_time = chrono::high_resolution_clock::now();
    int last_s = -1; int t = 0;
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
        int n_blocks = 16384 / best_b;
        for(int k=0; k < n_blocks * best_b; ++k) block_indices_gpu[k] = gen() % (H * INPUT_DIM);
        bool use_global = (best_b > 128);
        int smem_size = use_global ? 0 : (best_b * best_b + best_b) * sizeof(float);
        block_newton_kernel<<<n_blocks, 256, smem_size>>>(W1.data(), block_indices_gpu, batch_imgs.data(), dh_scaled.data(), global_H, global_g, 0.5f, H, BATCH_SIZE, best_b, use_global);
        t++; cudaDeviceSynchronize();
        int current_s = chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start_time).count();
        if (current_s > last_s) {
            float err = (1.0f - (*correct_gpu / (float)BATCH_SIZE)) * 100.0f;
            cout << "[Time: " << current_s << "s] Iter: " << t << " | Err: " << err << "% | Block: " << best_b << endl;
            last_s = current_s;
        }
    }
    cublasDestroy(handle); return 0;
}
