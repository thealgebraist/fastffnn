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

const size_t P = 256; // Population size
const size_t H = 1024;
const size_t D = 3072;
const size_t B = 50000;
const size_t C = 10;
const int TRAIN_LIMIT_S = 600;

__global__ void add_noise_kernel(float* batch, const float* base, const float* noise, size_t P, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < P * size) {
        size_t p = i / size;
        size_t idx = i % size;
        if (p == 0) {
            batch[i] = base[idx]; // Candidate 0 is exactly the current base state
        } else {
            batch[i] = base[idx] + noise[i];
        }
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
        size_t h = rem % H; // column-major: rem = b * H + h
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
        for(size_t c=0; c<C; ++c) {
            sum_e += expf((logits_batch[p * C * B + b * C + c] + b2_batch[p * C + c]) - max_l);
        }
        
        int label = labels[b];
        float prob = expf((logits_batch[p * C * B + b * C + label] + b2_batch[p * C + label]) - max_l) / sum_e;
        float loss = -logf(prob + 1e-9f);
        
        atomicAdd(&loss_batch[p], loss);
        
        float max_p = -1.0;
        int pred = 0;
        for(size_t c=0; c<C; ++c) {
            float p_val = expf((logits_batch[p * C * B + b * C + c] + b2_batch[p * C + c]) - max_l) / sum_e;
            if(p_val > max_p) { max_p = p_val; pred = c; }
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
    cout << "H200 Population-Based Simulated Annealing (P=" << P << ", Full 50k Batch)..." << endl;
    download_cifar10();
    vector<float> cpu_images; vector<uint8_t> cpu_labels;
    for(int i=1; i<=5; ++i) load_cifar("cifar-10-batches-bin/data_batch_" + to_string(i) + ".bin", cpu_images, cpu_labels);
    
    cublasHandle_t handle; CHECK_CUBLAS(cublasCreate(&handle));
    curandGenerator_t prng; CHECK_CURAND(curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(prng, 42));

    // Allocate Device Memory
    float *d_X; CHECK_CUDA(cudaMalloc(&d_X, D * B * sizeof(float)));
    uint8_t *d_labels; CHECK_CUDA(cudaMalloc(&d_labels, B * sizeof(uint8_t)));
    CHECK_CUDA(cudaMemcpy(d_X, cpu_images.data(), D * B * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_labels, cpu_labels.data(), B * sizeof(uint8_t), cudaMemcpyHostToDevice));

    float *W1_base, *b1_base, *W2_base, *b2_base;
    CHECK_CUDA(cudaMalloc(&W1_base, H * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b1_base, H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&W2_base, C * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b2_base, C * sizeof(float)));

    // Initialize base weights
    mt19937 gen(42); 
    float s1 = sqrtf(2.0f / (D + H)), s2 = sqrtf(2.0f / (H + C)); 
    normal_distribution<float> d1(0, s1), d2(0, s2);
    vector<float> h_W1(H * D), h_W2(C * H);
    for(int i=0; i<H*D; ++i) h_W1[i] = d1(gen);
    for(int i=0; i<C*H; ++i) h_W2[i] = d2(gen);
    CHECK_CUDA(cudaMemcpy(W1_base, h_W1.data(), H * D * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(W2_base, h_W2.data(), C * H * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(b1_base, 0, H * sizeof(float)));
    CHECK_CUDA(cudaMemset(b2_base, 0, C * sizeof(float)));

    // Allocate Batched Memory (~14GB for P=256)
    float *W1_batch, *b1_batch, *W2_batch, *b2_batch;
    CHECK_CUDA(cudaMalloc(&W1_batch, P * H * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b1_batch, P * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&W2_batch, P * C * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b2_batch, P * C * sizeof(float)));

    float *W1_noise, *b1_noise, *W2_noise, *b2_noise;
    CHECK_CUDA(cudaMalloc(&W1_noise, P * H * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b1_noise, P * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&W2_noise, P * C * H * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&b2_noise, P * C * sizeof(float)));

    float *hs_batch, *logits_batch;
    CHECK_CUDA(cudaMalloc(&hs_batch, P * H * B * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&logits_batch, P * C * B * sizeof(float)));

    float* loss_gpu; int* correct_gpu;
    CHECK_CUDA(cudaMallocManaged(&loss_gpu, P * sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&correct_gpu, P * sizeof(int)));

    float T = 1.0f;
    float min_T = 1e-4f;
    float cooling_rate = 0.995f;
    float current_loss = 1e10f;
    int t = 0;
    auto start_time = chrono::high_resolution_clock::now();
    int last_s = -1;

    while (chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start_time).count() < TRAIN_LIMIT_S) {
        float sigma = max(0.0001f, T * 0.05f); // Noise scales with Temperature

        // Generate Noise
        CHECK_CURAND(curandGenerateNormal(prng, W1_noise, P * H * D, 0.0f, sigma));
        CHECK_CURAND(curandGenerateNormal(prng, b1_noise, P * H, 0.0f, sigma));
        CHECK_CURAND(curandGenerateNormal(prng, W2_noise, P * C * H, 0.0f, sigma));
        CHECK_CURAND(curandGenerateNormal(prng, b2_noise, P * C, 0.0f, sigma));

        // Create Population
        add_noise_kernel<<<(P * H * D + 255) / 256, 256>>>(W1_batch, W1_base, W1_noise, P, H * D);
        add_noise_kernel<<<(P * H + 255) / 256, 256>>>(b1_batch, b1_base, b1_noise, P, H);
        add_noise_kernel<<<(P * C * H + 255) / 256, 256>>>(W2_batch, W2_base, W2_noise, P, C * H);
        add_noise_kernel<<<(P * C + 255) / 256, 256>>>(b2_batch, b2_base, b2_noise, P, C);

        // Forward Pass: hs = W1 * X
        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUBLAS(cublasSgemmStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            H, B, D,
            &alpha,
            W1_batch, H, H * D,
            d_X, D, 0, // stride=0 for broadcast
            &beta,
            hs_batch, H, H * B,
            P
        ));

        // LReLU + Bias
        batched_lrelu_bias_kernel<<<(P * H * B + 255) / 256, 256>>>(hs_batch, b1_batch, P, H, B);

        // Forward Pass: logits = W2 * hs
        CHECK_CUBLAS(cublasSgemmStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            C, B, H,
            &alpha,
            W2_batch, C, C * H,
            hs_batch, H, H * B, // unique stride per batch
            &beta,
            logits_batch, C, C * B,
            P
        ));

        // Loss and Accuracy
        CHECK_CUDA(cudaMemset(loss_gpu, 0, P * sizeof(float)));
        CHECK_CUDA(cudaMemset(correct_gpu, 0, P * sizeof(int)));
        batched_softmax_loss_kernel<<<(P * B + 255) / 256, 256>>>(logits_batch, b2_batch, d_labels, loss_gpu, correct_gpu, P, B, C);
        CHECK_CUDA(cudaDeviceSynchronize());

        // Find Best Candidate
        int best_p = 0;
        float best_loss = loss_gpu[0];
        for (int p = 1; p < P; ++p) {
            if (loss_gpu[p] < best_loss) {
                best_loss = loss_gpu[p];
                best_p = p;
            }
        }

        // Initialize current_loss on first iter
        if (t == 0) current_loss = loss_gpu[0];

        // Metropolis Acceptance
        bool accept = false;
        if (best_p != 0) {
            if (best_loss < current_loss) {
                accept = true;
            } else {
                float avg_diff = (best_loss - current_loss) / B;
                float prob = expf(-avg_diff / T);
                float rand_val = (float)rand() / RAND_MAX;
                if (rand_val < prob) accept = true;
            }
        }

        if (accept) {
            extract_best_kernel<<<(H * D + 255) / 256, 256>>>(W1_base, W1_batch, best_p, H * D);
            extract_best_kernel<<<(H + 255) / 256, 256>>>(b1_base, b1_batch, best_p, H);
            extract_best_kernel<<<(C * H + 255) / 256, 256>>>(W2_base, W2_batch, best_p, C * H);
            extract_best_kernel<<<(C + 255) / 256, 256>>>(b2_base, b2_batch, best_p, C);
            current_loss = best_loss;
        }

        if (T > min_T) T *= cooling_rate;
        t++;

        int current_s = chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start_time).count();
        if (current_s > last_s) {
            int display_p = accept ? best_p : 0;
            float err = (1.0f - (correct_gpu[display_p] / (float)B)) * 100.0f;
            cout << "[Time: " << current_s << "s] Iter: " << t << " | Err: " << err << "% | Avg Loss: " << current_loss / B << " | T: " << T << (accept ? " [ACCEPTED]" : " [REJECTED]") << endl;
            last_s = current_s;
        }
    }

    cublasDestroy(handle); curandDestroyGenerator(prng);
    return 0;
}
