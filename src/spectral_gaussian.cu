#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <random>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <sys/stat.h>

using namespace std;

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(1); \
    } \
}

#define CHECK_CUFFT(call) { \
    cufftResult err = call; \
    if(err != CUFFT_SUCCESS) { \
        cerr << "CUFFT error: " << err << " at " << __FILE__ << ":" << __LINE__ << endl; \
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

const int INPUT_DIM = 3072;
const int HIDDEN_DIM = 1024;
const int NUM_CLASSES = 10;
const int BATCH_SIZE = 50000;
const int TRAIN_LIMIT_S = 600;

__global__ void copy_to_complex_kernel(cufftComplex* complex_h, const float* real_h, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        complex_h[i].x = real_h[i];
        complex_h[i].y = 0.0f;
    }
}

__global__ void copy_from_complex_kernel(float* real_h, const cufftComplex* complex_h, int size, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        real_h[i] = complex_h[i].x * scale;
    }
}

__global__ void spectral_ode_step_kernel(cufftComplex* freq, const float2* filter, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float r = freq[i].x * filter[i % HIDDEN_DIM].x - freq[i].y * filter[i % HIDDEN_DIM].y;
        float im = freq[i].x * filter[i % HIDDEN_DIM].y + freq[i].y * filter[i % HIDDEN_DIM].x;
        freq[i].x = r;
        freq[i].y = im;
    }
}

__global__ void lrelu_kernel(float* h, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        h[i] = h[i] > 0 ? h[i] : 0.1f * h[i];
    }
}

__global__ void softmax_loss_kernel(const float* logits, const uint8_t* labels, float* dLogits, float* loss, int* correct, int B, int C) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) {
        float max_l = -1e10;
        for(int c=0; c<C; ++c) { float l = logits[b * C + c]; if(l > max_l) max_l = l; }
        float sum_e = 0;
        for(int c=0; c<C; ++c) sum_e += expf(logits[b * C + c] - max_l);
        int label = labels[b];
        float prob = expf(logits[b * C + label] - max_l) / sum_e;
        atomicAdd(loss, -logf(prob + 1e-9f));
        int pred = 0; float max_p = -1.0;
        for(int c=0; c<C; ++c) {
            float p = expf(logits[b * C + c] - max_l) / sum_e;
            dLogits[b * C + c] = p - (c == label ? 1.0f : 0.0f);
            if(p > max_p) { max_p = p; pred = c; }
        }
        if(pred == label) atomicAdd(correct, 1);
    }
}

__global__ void update_weights_kernel(float* W, const float* grad, float lr, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) W[i] -= lr * grad[i];
}

void download_cifar10() {
    struct stat buffer;
    if (stat("cifar-10-batches-bin/data_batch_1.bin", &buffer) != 0) {
        system("wget -qO- https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz | tar xz");
    }
}

void load_cifar(vector<float>& images, vector<uint8_t>& labels) {
    for (int b_idx = 1; b_idx <= 5; ++b_idx) {
        ifstream file("cifar-10-batches-bin/data_batch_" + to_string(b_idx) + ".bin", ios::binary);
        for (int i = 0; i < 10000; ++i) {
            uint8_t label; file.read((char*)&label, 1); labels.push_back(label);
            vector<uint8_t> raw(3072); file.read((char*)raw.data(), 3072);
            for(int p=0; p<3072; ++p) images.push_back(raw[p] / 255.0f);
        }
    }
}

int main() {
    cout << "CUDA Spectral ODE Neural Solver (H200 cuFFT)..." << endl;
    download_cifar10();
    vector<float> cpu_images; vector<uint8_t> cpu_labels;
    load_cifar(cpu_images, cpu_labels);

    float *d_X; uint8_t *d_labels;
    CHECK_CUDA(cudaMalloc(&d_X, BATCH_SIZE * INPUT_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_labels, BATCH_SIZE * sizeof(uint8_t)));
    CHECK_CUDA(cudaMemcpy(d_X, cpu_images.data(), BATCH_SIZE * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_labels, cpu_labels.data(), BATCH_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice));

    float *W1, *W2;
    CHECK_CUDA(cudaMalloc(&W1, HIDDEN_DIM * INPUT_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&W2, NUM_CLASSES * HIDDEN_DIM * sizeof(float)));
    
    mt19937 gen(42);
    normal_distribution<float> d1(0, sqrtf(2.0f/INPUT_DIM)), d2(0, sqrtf(2.0f/HIDDEN_DIM));
    vector<float> h_W1(HIDDEN_DIM * INPUT_DIM), h_W2(NUM_CLASSES * HIDDEN_DIM);
    for(float& w : h_W1) w = d1(gen);
    for(float& w : h_W2) w = d2(gen);
    CHECK_CUDA(cudaMemcpy(W1, h_W1.data(), HIDDEN_DIM * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(W2, h_W2.data(), NUM_CLASSES * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice));

    float2 *d_filter; CHECK_CUDA(cudaMalloc(&d_filter, HIDDEN_DIM * sizeof(float2)));
    vector<float2> h_filter(HIDDEN_DIM, {1.0f, 0.0f});
    CHECK_CUDA(cudaMemcpy(d_filter, h_filter.data(), HIDDEN_DIM * sizeof(float2), cudaMemcpyHostToDevice));

    cublasHandle_t handle; CHECK_CUBLAS(cublasCreate(&handle));
    cufftHandle fft_plan; CHECK_CUFFT(cufftPlan1d(&fft_plan, HIDDEN_DIM, CUFFT_C2C, BATCH_SIZE));

    float *d_h, *d_logits, *d_dLogits, *d_dW2;
    cufftComplex *d_h_complex;
    CHECK_CUDA(cudaMalloc(&d_h, BATCH_SIZE * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_h_complex, BATCH_SIZE * HIDDEN_DIM * sizeof(cufftComplex)));
    CHECK_CUDA(cudaMalloc(&d_logits, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dLogits, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dW2, NUM_CLASSES * HIDDEN_DIM * sizeof(float)));

    float* loss_gpu; int* correct_gpu;
    CHECK_CUDA(cudaMallocManaged(&loss_gpu, sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&correct_gpu, sizeof(int)));

    auto start_time = chrono::high_resolution_clock::now();
    float lr = 0.001f;
    int iter = 0;

    while (chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start_time).count() < TRAIN_LIMIT_S) {
        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, HIDDEN_DIM, BATCH_SIZE, INPUT_DIM, &alpha, W1, HIDDEN_DIM, d_X, INPUT_DIM, &beta, d_h, HIDDEN_DIM));

        copy_to_complex_kernel<<<(BATCH_SIZE * HIDDEN_DIM + 255) / 256, 256>>>(d_h_complex, d_h, BATCH_SIZE * HIDDEN_DIM);
        
        CHECK_CUFFT(cufftExecC2C(fft_plan, d_h_complex, d_h_complex, CUFFT_FORWARD));
        spectral_ode_step_kernel<<<(BATCH_SIZE * HIDDEN_DIM + 255) / 256, 256>>>(d_h_complex, d_filter, BATCH_SIZE * HIDDEN_DIM);
        CHECK_CUFFT(cufftExecC2C(fft_plan, d_h_complex, d_h_complex, CUFFT_INVERSE));

        copy_from_complex_kernel<<<(BATCH_SIZE * HIDDEN_DIM + 255) / 256, 256>>>(d_h, d_h_complex, BATCH_SIZE * HIDDEN_DIM, 1.0f / HIDDEN_DIM);
        lrelu_kernel<<<(BATCH_SIZE * HIDDEN_DIM + 255) / 256, 256>>>(d_h, BATCH_SIZE * HIDDEN_DIM);

        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, NUM_CLASSES, BATCH_SIZE, HIDDEN_DIM, &alpha, W2, NUM_CLASSES, d_h, HIDDEN_DIM, &beta, d_logits, NUM_CLASSES));

        *loss_gpu = 0; *correct_gpu = 0;
        softmax_loss_kernel<<<(BATCH_SIZE + 255) / 256, 256>>>(d_logits, d_labels, d_dLogits, loss_gpu, correct_gpu, BATCH_SIZE, NUM_CLASSES);
        CHECK_CUDA(cudaDeviceSynchronize());

        float g_alpha = 1.0f / BATCH_SIZE;
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, NUM_CLASSES, HIDDEN_DIM, BATCH_SIZE, &g_alpha, d_dLogits, NUM_CLASSES, d_h, HIDDEN_DIM, &beta, d_dW2, NUM_CLASSES));
        update_weights_kernel<<<(NUM_CLASSES * HIDDEN_DIM + 255) / 256, 256>>>(W2, d_dW2, lr, NUM_CLASSES * HIDDEN_DIM);

        iter++;
        int elapsed = chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start_time).count();
        if (iter % 10 == 0) {
            cout << "[Time: " << elapsed << "s] Iter: " << iter << " | Err: " << (1.0f - (*correct_gpu / (float)BATCH_SIZE)) * 100.0f << "%" << endl;
        }
    }

    cublasDestroy(handle); cufftDestroy(fft_plan);
    return 0;
}
