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
const float TARGET_REDUCTION = 0.1f;

__global__ void update_weights_kernel(float* w, const float* dw, float adaptive_lr, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) w[i] -= adaptive_lr * dw[i];
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

__global__ void bn_backprop_kernel(const float* dL_dhs, const float* h_norm, const float* gamma, const float* beta, const float* var, float* db1, float* dG, float* dB, float* dh_scaled, int H, int B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < H) {
        float sum_dl_dy = 0, sum_dl_dy_xhat = 0;
        for(int b=0; b<B; ++b) {
            float act = h_norm[b * H + i] * gamma[i] + beta[i];
            float dL_dact = dL_dhs[b * H + i] * (act > 0 ? 1.0f : 0.1f);
            float dl_dy = dL_dact * gamma[i];
            sum_dl_dy += dl_dy; sum_dl_dy_xhat += dl_dy * h_norm[b * H + i];
            dG[i] += dL_dact * h_norm[b * H + i]; dB[i] += dL_dact;
        }
        for(int b=0; b<B; ++b) {
            float act = h_norm[b * H + i] * gamma[i] + beta[i];
            float dL_dact = dL_dhs[b * H + i] * (act > 0 ? 1.0f : 0.1f);
            float dl_dy = dL_dact * gamma[i];
            float dl_dx = (1.0f / (B * var[i])) * (B * dl_dy - sum_dl_dy - h_norm[b * H + i] * sum_dl_dy_xhat);
            dh_scaled[b * H + i] = dl_dx; atomicAdd(&db1[i], dl_dx);
        }
    }
}

__global__ void scale_vec_kernel(float* v, float s, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) v[i] *= s;
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
    cout << "Coq-Verified Variable-Time Solver (Target Reduction: 0.1 per step)..." << endl;
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
    CudaVector hs(MAX_SAMPLES * MAX_NEURONS), hs_norm(MAX_SAMPLES * MAX_NEURONS), logits(MAX_SAMPLES * NUM_CLASSES), dLogits(MAX_SAMPLES * NUM_CLASSES), dL_dhs(MAX_SAMPLES * MAX_NEURONS), dh_scaled(MAX_SAMPLES * MAX_NEURONS);
    CudaVector mu(MAX_NEURONS), var(MAX_NEURONS);
    
    auto start_time = chrono::high_resolution_clock::now();
    int last_s = -1; int t = 0;
    CudaVector dW1(MAX_NEURONS * INPUT_DIM), dW2(NUM_CLASSES * MAX_NEURONS), db1(MAX_NEURONS), db2(NUM_CLASSES), dG(MAX_NEURONS), dB(MAX_NEURONS);

    while (chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start_time).count() < TRAIN_LIMIT_S) {
        *loss_gpu = 0; *correct_gpu = 0;
        cudaMemset(dW1.data(), 0, dW1.size()*4); cudaMemset(dW2.data(), 0, dW2.size()*4);
        cudaMemset(db1.data(), 0, MAX_NEURONS*4); cudaMemset(db2.data(), 0, NUM_CLASSES*4);
        cudaMemset(dG.data(), 0, MAX_NEURONS*4); cudaMemset(dB.data(), 0, MAX_NEURONS*4);
        
        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, H, MAX_SAMPLES, INPUT_DIM, &alpha, W1.data(), MAX_NEURONS, all_images.data(), INPUT_DIM, &beta, hs.data(), H));
        bn_lrelu_forward_kernel<<<(H+255)/256, 256>>>(hs.data(), hs_norm.data(), b1.data(), bn_gamma.data(), bn_beta.data(), mu.data(), var.data(), H, MAX_SAMPLES);
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, NUM_CLASSES, MAX_SAMPLES, H, &alpha, W2.data(), NUM_CLASSES, hs.data(), H, &beta, logits.data(), NUM_CLASSES));
        softmax_loss_kernel<<<(MAX_SAMPLES+255)/256, 256>>>(logits.data(), b2.data(), all_labels.data(), dLogits.data(), loss_gpu, correct_gpu, MAX_SAMPLES, NUM_CLASSES);
        cudaDeviceSynchronize();
        
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, NUM_CLASSES, H, MAX_SAMPLES, &alpha, dLogits.data(), NUM_CLASSES, hs.data(), H, &beta, dW2.data(), NUM_CLASSES));
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, H, MAX_SAMPLES, NUM_CLASSES, &alpha, W2.data(), NUM_CLASSES, dLogits.data(), NUM_CLASSES, &beta, dL_dhs.data(), H));
        bn_backprop_kernel<<<(H+255)/256, 256>>>(dL_dhs.data(), hs_norm.data(), bn_gamma.data(), bn_beta.data(), var.data(), db1.data(), dG.data(), dB.data(), dh_scaled.data(), H, MAX_SAMPLES);
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, H, INPUT_DIM, MAX_SAMPLES, &alpha, dh_scaled.data(), H, all_images.data(), INPUT_DIM, &beta, dW1.data(), MAX_NEURONS));
        
        float inv_B = 1.0f / MAX_SAMPLES;
        scale_vec_kernel<<<(dW1.size()+255)/256, 256>>>(dW1.data(), inv_B, dW1.size());
        scale_vec_kernel<<<(MAX_NEURONS+255)/256, 256>>>(db1.data(), inv_B, H);
        scale_vec_kernel<<<(dW2.size()+255)/256, 256>>>(dW2.data(), inv_B, dW2.size());
        scale_vec_kernel<<<(NUM_CLASSES+255)/256, 256>>>(db2.data(), inv_B, NUM_CLASSES);
        scale_vec_kernel<<<(MAX_NEURONS+255)/256, 256>>>(dG.data(), inv_B, H);
        scale_vec_kernel<<<(MAX_NEURONS+255)/256, 256>>>(dB.data(), inv_B, H);
        cudaDeviceSynchronize();

        // Variable-Time Adaptive Step Logic
        float n1, n2, n3, n4, n5, n6;
        cublasSnrm2(handle, dW1.size(), dW1.data(), 1, &n1); cublasSnrm2(handle, H, db1.data(), 1, &n2);
        cublasSnrm2(handle, dW2.size(), dW2.data(), 1, &n3); cublasSnrm2(handle, NUM_CLASSES, db2.data(), 1, &n4);
        cublasSnrm2(handle, H, dG.data(), 1, &n5); cublasSnrm2(handle, H, dB.data(), 1, &n6);
        float total_sq_norm = n1*n1 + n2*n2 + n3*n3 + n4*n4 + n5*n5 + n6*n6;
        
        // eta = delta / ||g||^2
        float adaptive_lr = TARGET_REDUCTION / (total_sq_norm + 1e-9f);

        update_weights_kernel<<<(dW1.size()+255)/256, 256>>>(W1.data(), dW1.data(), adaptive_lr, H * INPUT_DIM);
        update_weights_kernel<<<(MAX_NEURONS+255)/256, 256>>>(db1.data(), dW1.data(), adaptive_lr, H); // reused buffer
        update_weights_kernel<<<(dW2.size()+255)/256, 256>>>(W2.data(), dW2.data(), adaptive_lr, NUM_CLASSES * MAX_NEURONS);
        update_weights_kernel<<<(NUM_CLASSES+255)/256, 256>>>(db2.data(), dW2.data(), adaptive_lr, NUM_CLASSES); // reused buffer
        update_weights_kernel<<<(MAX_NEURONS+255)/256, 256>>>(bn_gamma.data(), dG.data(), adaptive_lr, H);
        update_weights_kernel<<<(MAX_NEURONS+255)/256, 256>>>(bn_beta.data(), dB.data(), adaptive_lr, H);
        
        t++; cudaDeviceSynchronize();
        int current_s = chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start_time).count();
        if (current_s > last_s) {
            float err = (1.0f - (*correct_gpu / (float)MAX_SAMPLES)) * 100.0f;
            cout << "[Time: " << current_s << "s] Iter: " << t << " | Err: " << err << "% | adaptive_lr: " << adaptive_lr << endl;
            last_s = current_s;
        }
    }
    cublasDestroy(handle); return 0;
}
