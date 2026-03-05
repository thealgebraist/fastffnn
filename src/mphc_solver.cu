#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cublas_v2.h>
#include <curand.h>
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

#define CHECK_CURAND(call) { \
    curandStatus_t status = call; \
    if(status != CURAND_STATUS_SUCCESS) { \
        cerr << "CURAND error at " << __FILE__ << ":" << __LINE__ << endl; \
        exit(1); \
    } \
}

const size_t P = 600; // Population size for Massive Parallel Hill Climbing
const size_t H = 1024;
const size_t D = 3072;
const size_t B = 50000;
const size_t C = 10;
const int TRAIN_LIMIT_S = 600;

__global__ void add_base_to_noise_kernel(float* batch, const float* base, size_t P, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < P * size) {
        size_t p = i / size;
        size_t idx = i % size;
        if (p == 0) batch[i] = base[idx]; // First candidate is current state
        else batch[i] += base[idx];
    }
}

__global__ void extract_best_kernel(float* base, const float* batch, size_t best_p, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        base[i] = batch[best_p * size + i];
    }
}

__global__ void batched_lrelu_bias_kernel(float* hs_batch, const float* b1_batch, size_t P, size_t H, size_t B) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < P * B * H) {
        size_t p = idx / (B * H);
        size_t rem = idx % (B * H);
        size_t h = rem % H;
        float val = hs_batch[idx] + b1_batch[p * H + h];
        hs_batch[idx] = val > 0 ? val : 0.1f * val;
    }
}

__global__ void batched_softmax_loss_kernel(const float* logits_batch, const float* b2_batch, const uint8_t* labels, float* loss_batch, int* correct_batch, size_t P, size_t B, size_t C) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < P * B) {
        size_t p = idx / B;
        size_t b = idx % B;
        float max_l = -1e10;
        for(size_t c=0; c<C; ++c) {
            float l = logits_batch[p * C * B + b * C + c] + b2_batch[p * C + c];
            if(l > max_l) max_l = l;
        }
        float sum_e = 0;
        for(size_t c=0; c<C; ++c) sum_e += expf((logits_batch[p * C * B + b * C + c] + b2_batch[p * C + c]) - max_l);
        int label = labels[b];
        float prob = expf((logits_batch[p * C * B + b * C + label] + b2_batch[p * C + label]) - max_l) / sum_e;
        atomicAdd(&loss_batch[p], -logf(prob + 1e-9f));
        float max_p = -1.0; int pred = 0;
        for(size_t c=0; c<C; ++c) {
            float pv = expf((logits_batch[p * C * B + b * C + c] + b2_batch[p * C + c]) - max_l) / sum_e;
            if(pv > max_p) { max_p = pv; pred = c; }
        }
        if (pred == label) atomicAdd(&correct_batch[p], 1);
    }
}

void download_cifar10() {
    struct stat buffer;
    if (stat("cifar-10-batches-bin/data_batch_1.bin", &buffer) != 0) {
        system("wget -qO- https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz | tar xz");
    }
}

bool load_cifar(const string& path, vector<float>& images, vector<uint8_t>& labels) {
    ifstream file(path, ios::binary); if (!file) return false;
    for (int i = 0; i < 10000; ++i) {
        uint8_t label; file.read((char*)&label, 1); labels.push_back(label);
        vector<uint8_t> raw(3072); file.read((char*)raw.data(), 3072);
        for(int p=0; p<3072; ++p) images.push_back(raw[p] / 255.0f);
    }
    return true;
}

int main() {
    cout << "Massive Parallel Hill Climbing (P=" << P << ", Full 50k Batch) H200 optimized..." << endl;
    download_cifar10();
    vector<float> cpu_images; vector<uint8_t> cpu_labels;
    for(int i=1; i<=5; ++i) load_cifar("cifar-10-batches-bin/data_batch_" + to_string(i) + ".bin", cpu_images, cpu_labels);
    
    cublasHandle_t handle; CHECK_CUBLAS(cublasCreate(&handle));
    curandGenerator_t prng; CHECK_CURAND(curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(prng, 42));

    float *d_X; CHECK_CUDA(cudaMalloc(&d_X, D * B * sizeof(float)));
    uint8_t *d_labels; CHECK_CUDA(cudaMalloc(&d_labels, B * sizeof(uint8_t)));
    CHECK_CUDA(cudaMemcpy(d_X, cpu_images.data(), D * B * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_labels, cpu_labels.data(), B * sizeof(uint8_t), cudaMemcpyHostToDevice));

    float *W1_base, *b1_base, *W2_base, *b2_base;
    CHECK_CUDA(cudaMalloc(&W1_base, H * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b1_base, H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&W2_base, C * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b2_base, C * sizeof(float)));

    mt19937 gen(42); vector<float> h_W1(H * D), h_W2(C * H);
    float s1 = sqrtf(2.0f / (D + H)), s2 = sqrtf(2.0f / (H + C));
    normal_distribution<float> d1(0, s1), d2(0, s2);
    for(int i=0; i<H*D; ++i) h_W1[i] = d1(gen);
    for(int i=0; i<C*H; ++i) h_W2[i] = d2(gen);
    CHECK_CUDA(cudaMemcpy(W1_base, h_W1.data(), H * D * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(W2_base, h_W2.data(), C * H * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(b1_base, 0, H * sizeof(float)));
    CHECK_CUDA(cudaMemset(b2_base, 0, C * sizeof(float)));

    float *W1_batch, *b1_batch, *W2_batch, *b2_batch;
    CHECK_CUDA(cudaMalloc(&W1_batch, P * H * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b1_batch, P * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&W2_batch, P * C * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b2_batch, P * C * sizeof(float)));

    float *hs_batch, *logits_batch;
    CHECK_CUDA(cudaMalloc(&hs_batch, P * H * B * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&logits_batch, P * C * B * sizeof(float)));

    float* loss_gpu; int* correct_gpu;
    CHECK_CUDA(cudaMallocManaged(&loss_gpu, P * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&correct_gpu, P * sizeof(int)));

    float current_loss = 1e10f;
    float noise_scale = 0.01f;
    int t = 0;
    auto start_time = chrono::high_resolution_clock::now();
    int last_s = -1;

    while (chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start_time).count() < TRAIN_LIMIT_S) {
        // Step 1: Generate Noise directly into batch
        CHECK_CURAND(curandGenerateNormal(prng, W1_batch, P * H * D, 0.0f, noise_scale));
        CHECK_CURAND(curandGenerateNormal(prng, b1_batch, P * H, 0.0f, noise_scale));
        CHECK_CURAND(curandGenerateNormal(prng, W2_batch, P * C * H, 0.0f, noise_scale));
        CHECK_CURAND(curandGenerateNormal(prng, b2_batch, P * C, 0.0f, noise_scale));

        // Step 2: W_batch = W_noise + W_base
        add_base_to_noise_kernel<<<(P * H * D + 255) / 256, 256>>>(W1_batch, W1_base, P, H * D);
        add_base_to_noise_kernel<<<(P * H + 255) / 256, 256>>>(b1_batch, b1_base, P, H);
        add_base_to_noise_kernel<<<(P * C * H + 255) / 256, 256>>>(W2_batch, W2_base, P, C * H);
        add_base_to_noise_kernel<<<(P * C + 255) / 256, 256>>>(b2_batch, b2_base, P, C);

        // Step 3: Batched Forward Pass
        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUBLAS(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, H, B, D, &alpha, W1_batch, H, H * D, d_X, D, 0, &beta, hs_batch, H, H * B, P));
        batched_lrelu_bias_kernel<<<(P * H * B + 255) / 256, 256>>>(hs_batch, b1_batch, P, H, B);
        CHECK_CUBLAS(cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, C, B, H, &alpha, W2_batch, C, C * H, hs_batch, H, H * B, &beta, logits_batch, C, C * B, P));

        // Step 4: Batch Loss/Correct calculation
        CHECK_CUDA(cudaMemset(loss_gpu, 0, P * sizeof(float)));
        CHECK_CUDA(cudaMemset(correct_gpu, 0, P * sizeof(int)));
        batched_softmax_loss_kernel<<<(P * B + 255) / 256, 256>>>(logits_batch, b2_batch, d_labels, loss_gpu, correct_gpu, P, B, C);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Step 5: Find Best Candidate
        int best_p = 0; float best_l = loss_gpu[0];
        for(int p=1; p<P; ++p) { if(loss_gpu[p] < best_l) { best_l = loss_gpu[p]; best_p = p; } }

        if (t == 0) current_loss = loss_gpu[0];

        // Step 6: Hill Climbing Step (Only accept if strictly better)
        if (best_p != 0 && best_l < current_loss) {
            extract_best_kernel<<<(H * D + 255) / 256, 256>>>(W1_base, W1_batch, best_p, H * D);
            extract_best_kernel<<<(H + 255) / 256, 256>>>(b1_base, b1_batch, best_p, H);
            extract_best_kernel<<<(C * H + 255) / 256, 256>>>(W2_base, W2_batch, best_p, C * H);
            extract_best_kernel<<<(C + 255) / 256, 256>>>(b2_base, b2_batch, best_p, C);
            current_loss = best_l;
            noise_scale *= 1.01f; // Slight expansion on success
        } else {
            noise_scale *= 0.99f; // Refinement on failure
        }

        t++;
        int current_s = chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start_time).count();
        if (current_s > last_s) {
            float err = (1.0f - (correct_gpu[best_p] / (float)B)) * 100.0f;
            cout << "[Time: " << current_s << "s] Iter: " << t << " | Err: " << err << "% | Loss: " << current_loss / B << " | Sigma: " << noise_scale << (best_p != 0 && best_l < current_loss ? " [CLIMB]" : "") << endl;
            last_s = current_s;
        }
    }
    cublasDestroy(handle); curandDestroyGenerator(prng); return 0;
}
