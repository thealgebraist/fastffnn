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
const int MAX_NEURONS = 8192;
const int NUM_CLASSES = 10;
const int TRAIN_LIMIT_S = 600; 
const int INITIAL_NEURONS = 512;
const int MAX_SAMPLES = 50000;
const float CLIP_THRESHOLD = 5.0f;

__global__ void add_eye_kernel(float* A, float lambda, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i * n + i] += lambda;
}

__global__ void update_weights_kernel(float* w, const float* dw, float lr, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) w[i] -= lr * dw[i];
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

void invert_matrix_cusolver(cusolverDnHandle_t handle, float* A, float* A_inv, int n, float lambda) {
    add_eye_kernel<<<(n+255)/256, 256>>>(A, lambda, n);
    int work_size = 0;
    CHECK_CUSOLVER(cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, n, A, n, &work_size));
    float* work; cudaMalloc(&work, work_size * sizeof(float));
    int* info; cudaMallocManaged(&info, sizeof(int));
    
    // Identity for inverse
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
    cout << "Pure Full-Batch K-FAC (Gauss-Newton) Solver with cuSOLVER (H200)..." << endl;
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
    cusolverDnHandle_t solver_handle; CHECK_CUSOLVER(cusolverDnCreate(&solver_handle));
    float* loss_gpu; int* correct_gpu; CHECK_CUDA(cudaMallocManaged(&loss_gpu, sizeof(float))); CHECK_CUDA(cudaMallocManaged(&correct_gpu, sizeof(int)));
    
    CudaVector hs(MAX_SAMPLES * MAX_NEURONS), hs_norm(MAX_SAMPLES * MAX_NEURONS), logits(MAX_SAMPLES * NUM_CLASSES), dLogits(MAX_SAMPLES * NUM_CLASSES), dL_dhs(MAX_SAMPLES * MAX_NEURONS), dh_scaled(MAX_SAMPLES * MAX_NEURONS);
    CudaVector mu(MAX_NEURONS), var(MAX_NEURONS);
    CudaVector A1(INPUT_DIM * INPUT_DIM), G1(MAX_NEURONS * MAX_NEURONS), A2(MAX_NEURONS * MAX_NEURONS), G2(NUM_CLASSES * NUM_CLASSES);
    CudaVector A1_inv(INPUT_DIM * INPUT_DIM), G1_inv(MAX_NEURONS * MAX_NEURONS), A2_inv(MAX_NEURONS * MAX_NEURONS), G2_inv(NUM_CLASSES * NUM_CLASSES);
    CudaVector temp1(MAX_NEURONS * INPUT_DIM), temp2(NUM_CLASSES * MAX_NEURONS);

    auto start_time = chrono::high_resolution_clock::now();
    int last_s = -1; float lr = 0.1f; int t = 0;
    CudaVector dW1(MAX_NEURONS * INPUT_DIM), dW2(NUM_CLASSES * MAX_NEURONS), db1(MAX_NEURONS), db2(NUM_CLASSES), dG(MAX_NEURONS), dB(MAX_NEURONS);

    while (chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start_time).count() < TRAIN_LIMIT_S) {
        *loss_gpu = 0; *correct_gpu = 0;
        cudaMemset(dW1.data(), 0, dW1.size()*4); cudaMemset(dW2.data(), 0, dW2.size()*4);
        
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
        scale_vec_kernel<<<(dW2.size()+255)/256, 256>>>(dW2.data(), inv_B, dW2.size());

        float gn_alpha = 1.0f / MAX_SAMPLES;
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, INPUT_DIM, INPUT_DIM, MAX_SAMPLES, &gn_alpha, all_images.data(), INPUT_DIM, all_images.data(), INPUT_DIM, &beta, A1.data(), INPUT_DIM));
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, H, H, MAX_SAMPLES, &gn_alpha, dh_scaled.data(), H, dh_scaled.data(), H, &beta, G1.data(), H));
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, H, H, MAX_SAMPLES, &gn_alpha, hs.data(), H, hs.data(), H, &beta, A2.data(), H));
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, NUM_CLASSES, NUM_CLASSES, MAX_SAMPLES, &gn_alpha, dLogits.data(), NUM_CLASSES, dLogits.data(), NUM_CLASSES, &beta, G2.data(), NUM_CLASSES));
        
        invert_matrix_cusolver(solver_handle, A1.data(), A1_inv.data(), INPUT_DIM, 0.01f);
        invert_matrix_cusolver(solver_handle, G1.data(), G1_inv.data(), H, 0.01f);
        invert_matrix_cusolver(solver_handle, A2.data(), A2_inv.data(), H, 0.01f);
        invert_matrix_cusolver(solver_handle, G2.data(), G2_inv.data(), NUM_CLASSES, 0.01f);

        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, H, INPUT_DIM, H, &alpha, G1_inv.data(), H, dW1.data(), MAX_NEURONS, &beta, temp1.data(), H));
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, H, INPUT_DIM, INPUT_DIM, &alpha, temp1.data(), H, A1_inv.data(), INPUT_DIM, &beta, dW1.data(), MAX_NEURONS));
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, NUM_CLASSES, H, NUM_CLASSES, &alpha, G2_inv.data(), NUM_CLASSES, dW2.data(), NUM_CLASSES, &beta, temp2.data(), NUM_CLASSES));
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, NUM_CLASSES, H, H, &alpha, temp2.data(), NUM_CLASSES, A2_inv.data(), H, &beta, dW2.data(), NUM_CLASSES));
        
        update_weights_kernel<<<(dW1.size()+255)/256, 256>>>(W1.data(), dW1.data(), lr, H * INPUT_DIM);
        update_weights_kernel<<<(dW2.size()+255)/256, 256>>>(W2.data(), dW2.data(), lr, NUM_CLASSES * MAX_NEURONS);
        
        t++; cudaDeviceSynchronize();
        int current_s = chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start_time).count();
        if (current_s > last_s) {
            float err = (1.0f - (*correct_gpu / (float)MAX_SAMPLES)) * 100.0f;
            cout << "[Time: " << current_s << "s] Iter: " << t << " | Err: " << err << "% | Loss: " << *loss_gpu/MAX_SAMPLES << " [KFAC-PURE]" << endl;
            last_s = current_s;
        }
    }
    cublasDestroy(handle); cusolverDnDestroy(solver_handle); return 0;
}
