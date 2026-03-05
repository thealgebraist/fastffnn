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

const int IMG_SIZE = 32;
const int CHANNELS = 3;
const int INPUT_DIM = 3072;
const int MAX_NEURONS = 4096;
const int NUM_CLASSES = 10;
const int TRAIN_LIMIT_S = 600; 
const int INITIAL_IMAGES = 8192;
const int INITIAL_NEURONS = 64;

__global__ void fused_radam_kernel(float* w, float* m, float* v, const float* g, 
                                  float b1, float b2, float eps, float lr, 
                                  float b1_t, float b2_t, float rho_inf, float rho_t, 
                                  int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float grad = g[i];
        m[i] = b1 * m[i] + (1.0f - b1) * grad;
        v[i] = b2 * v[i] + (1.0f - b2) * grad * grad;
        float m_hat = m[i] / (1.0f - b1_t);
        if (rho_t > 5.0f) {
            float v_hat = sqrtf(v[i] / (1.0f - b2_t));
            float r_t = sqrtf(((rho_t - 4.0f) * (rho_t - 2.0f) * rho_inf) / ((rho_inf - 4.0f) * (rho_inf - 2.0f) * rho_t));
            w[i] -= lr * r_t * m_hat / (v_hat + eps);
        } else {
            w[i] -= lr * m_hat;
        }
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

void radam_call(float* w, float* m, float* v, const float* g, int t, float lr, int size) {
    if (size <= 0) return;
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float rho_inf = 2.0f / (1.0f - b2) - 1.0f;
    float b1_t = powf(b1, (float)t); float b2_t = powf(b2, (float)t);
    float rho_t = rho_inf - 2.0f * t * b2_t / (1.0f - b2_t);
    fused_radam_kernel<<<(size + 255) / 256, 256>>>(w, m, v, g, b1, b2, eps, lr, b1_t, b2_t, rho_inf, rho_t, size);
}

void download_cifar10() {
    struct stat buffer;
    if (stat("cifar-10-batches-bin/data_batch_1.bin", &buffer) != 0) {
        system("wget -qO- https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz | tar xz");
    }
}

bool load_cifar(const string& path, CudaVector& images, vector<uint8_t>& labels) {
    ifstream file(path, ios::binary); if (!file) return false;
    for (int i = 0; i < 10000; ++i) {
        uint8_t label; file.read((char*)&label, 1); labels.push_back(label);
        vector<uint8_t> raw(3072); file.read((char*)raw.data(), 3072);
        for(int p=0; p<3072; ++p) images.push_back(raw[p] / 255.0f);
    }
    return true;
}

int main() {
    cout << "Startup: Power-Scaling Benchmark (10s)..." << endl;
    download_cifar10();
    CudaVector all_images; vector<uint8_t> all_labels;
    for(int i=1; i<=5; ++i) load_cifar("cifar-10-batches-bin/data_batch_" + to_string(i) + ".bin", all_images, all_labels);
    
    int H = INITIAL_NEURONS;
    CudaVector W1(MAX_NEURONS * INPUT_DIM), b1(MAX_NEURONS, 0), W2(NUM_CLASSES * MAX_NEURONS), b2(NUM_CLASSES, 0), bn_gamma(MAX_NEURONS, 1.0f), bn_beta(MAX_NEURONS, 0.0f);
    CudaVector mW1(W1.size(), 0), vW1(W1.size(), 0), mb1(MAX_NEURONS, 0), vb1(MAX_NEURONS, 0), mW2(W2.size(), 0), vW2(W2.size(), 0), mb2(NUM_CLASSES, 0), vb2(NUM_CLASSES, 0), mG(MAX_NEURONS, 0), vG(MAX_NEURONS, 0), mB(MAX_NEURONS, 0), vB(MAX_NEURONS, 0);
    
    mt19937 gen(42); float s1 = sqrtf(2.0f / (INPUT_DIM + H)), s2 = sqrtf(2.0f / (H + NUM_CLASSES)); 
    normal_distribution<float> d1(0, s1), d2(0, s2);
    for(int i=0; i<MAX_NEURONS*INPUT_DIM; ++i) W1[i] = d1(gen);
    for(int i=0; i<NUM_CLASSES*MAX_NEURONS; ++i) W2[i] = d2(gen);

    cublasHandle_t handle; CHECK_CUBLAS(cublasCreate(&handle));
    vector<int> indices(all_labels.size()); iota(indices.begin(), indices.end(), 0); shuffle(indices.begin(), indices.end(), gen);
    vector<int> current_subset(indices.begin(), indices.begin() + INITIAL_IMAGES);
    
    int best_batch = 64; float best_efficiency = -1.0f;
    vector<int> candidate_batches = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    
    float* loss_gpu; int* correct_gpu; CHECK_CUDA(cudaMallocManaged(&loss_gpu, sizeof(float))); CHECK_CUDA(cudaMallocManaged(&correct_gpu, sizeof(int)));
    uint8_t* labels_gpu; CHECK_CUDA(cudaMallocManaged(&labels_gpu, 16384));
    CudaVector batch_imgs(16384 * INPUT_DIM), hs(16384 * MAX_NEURONS), hs_norm(16384 * MAX_NEURONS), logits(16384 * NUM_CLASSES), dLogits(16384 * NUM_CLASSES), dL_dhs(16384 * MAX_NEURONS), dh_scaled(16384 * MAX_NEURONS);
    CudaVector mu(MAX_NEURONS), var(MAX_NEURONS);

    for(int b_size : candidate_batches) {
        auto b_start = chrono::high_resolution_clock::now();
        float start_loss = 0;
        for(int i=0; i<5; ++i) {
            *loss_gpu = 0;
            for(int b=0; b<b_size; ++b) {
                int idx = current_subset[gen() % current_subset.size()];
                memcpy(&batch_imgs[b * INPUT_DIM], &all_images[idx * 3072], 3072 * sizeof(float));
                labels_gpu[b] = all_labels[idx];
            }
            float alpha = 1.0f, beta = 0.0f;
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, H, b_size, INPUT_DIM, &alpha, W1.data(), MAX_NEURONS, batch_imgs.data(), INPUT_DIM, &beta, hs.data(), H);
            bn_lrelu_forward_kernel<<<(H+255)/256, 256>>>(hs.data(), hs_norm.data(), b1.data(), bn_gamma.data(), bn_beta.data(), mu.data(), var.data(), H, b_size);
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, NUM_CLASSES, b_size, H, &alpha, W2.data(), NUM_CLASSES, hs.data(), H, &beta, logits.data(), NUM_CLASSES);
            softmax_loss_kernel<<<(b_size+255)/256, 256>>>(logits.data(), b2.data(), labels_gpu, dLogits.data(), loss_gpu, correct_gpu, b_size, NUM_CLASSES);
            cudaDeviceSynchronize();
            if(i==0) start_loss = *loss_gpu / b_size;
        }
        auto b_end = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(b_end - b_start).count();
        float efficiency = (start_loss > 0) ? (b_size / elapsed) : 0;
        cout << "[Bench] Batch: " << b_size << " | Efficiency: " << efficiency << " img/s" << endl;
        if(efficiency > best_efficiency) { best_efficiency = efficiency; best_batch = b_size; }
    }
    cout << "Winner: Batch Size " << best_batch << endl;

    auto start_time = chrono::high_resolution_clock::now();
    auto last_print_time = start_time;
    float lr = 0.001f, last_loss = 1e10;
    int stagnate_count = 0, t = 0, next_idx = INITIAL_IMAGES;
    CudaVector dW1(MAX_NEURONS * INPUT_DIM), db1(MAX_NEURONS), dW2(NUM_CLASSES * MAX_NEURONS), db2(NUM_CLASSES), dG(MAX_NEURONS), dB(MAX_NEURONS);

    while (chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start_time).count() < TRAIN_LIMIT_S) {
        *loss_gpu = 0; *correct_gpu = 0;
        cudaMemset(dW1.data(), 0, dW1.size()*4); cudaMemset(db1.data(), 0, MAX_NEURONS*4);
        cudaMemset(dW2.data(), 0, dW2.size()*4); cudaMemset(db2.data(), 0, NUM_CLASSES*4);
        cudaMemset(dG.data(), 0, MAX_NEURONS*4); cudaMemset(dB.data(), 0, MAX_NEURONS*4);
        for(int b=0; b<best_batch; ++b) {
            int idx = current_subset[gen() % current_subset.size()];
            batch_labels[b] = all_labels[idx]; memcpy(&batch_imgs[b * INPUT_DIM], &all_images[idx * 3072], 3072 * sizeof(float));
        }
        float alpha = 1.0f, beta = 0.0f;
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, H, best_batch, INPUT_DIM, &alpha, W1.data(), MAX_NEURONS, batch_imgs.data(), INPUT_DIM, &beta, hs.data(), H));
        bn_lrelu_forward_kernel<<<(H+255)/256, 256>>>(hs.data(), hs_norm.data(), b1.data(), bn_gamma.data(), bn_beta.data(), mu.data(), var.data(), H, best_batch);
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, NUM_CLASSES, best_batch, H, &alpha, W2.data(), NUM_CLASSES, hs.data(), H, &beta, logits.data(), NUM_CLASSES));
        softmax_loss_kernel<<<(best_batch+255)/256, 256>>>(logits.data(), b2.data(), batch_labels.data(), dLogits.data(), loss_gpu, correct_gpu, best_batch, NUM_CLASSES);
        cudaDeviceSynchronize();
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, NUM_CLASSES, H, best_batch, &alpha, dLogits.data(), NUM_CLASSES, hs.data(), H, &beta, dW2.data(), NUM_CLASSES));
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, H, best_batch, NUM_CLASSES, &alpha, W2.data(), NUM_CLASSES, dLogits.data(), NUM_CLASSES, &beta, dL_dhs.data(), H));
        bn_backprop_kernel<<<(H+255)/256, 256>>>(dL_dhs.data(), hs_norm.data(), bn_gamma.data(), bn_beta.data(), var.data(), db1.data(), dG.data(), dB.data(), dh_scaled.data(), H, best_batch);
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, H, INPUT_DIM, best_batch, &alpha, dh_scaled.data(), H, batch_imgs.data(), INPUT_DIM, &beta, dW1.data(), MAX_NEURONS));
        float avg_loss = *loss_gpu / best_batch; t++;
        radam_call(W1.data(), mW1.data(), vW1.data(), dW1.data(), t, lr, H * INPUT_DIM);
        radam_call(b1.data(), mb1.data(), vb1.data(), db1.data(), t, lr, H);
        radam_call(W2.data(), mW2.data(), vW2.data(), dW2.data(), t, lr, NUM_CLASSES * MAX_NEURONS);
        radam_call(b2.data(), mb2.data(), vb2.data(), db2.data(), t, lr, NUM_CLASSES);
        radam_call(bn_gamma.data(), mG.data(), vG.data(), dG.data(), t, lr, H);
        radam_call(bn_beta.data(), mB.data(), vB.data(), dB.data(), t, lr, H);
        cudaDeviceSynchronize();
        if (chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - last_print_time).count() >= 1) {
            cout << "[Time: " << (int)chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start_time).count() << "s] Err: " << (1.0f - (*correct_gpu / (float)best_batch)) * 100.0f << "% | Loss: " << avg_loss << " | Subset: " << current_subset.size() << " | Neurons: " << H << endl;
            last_print_time = chrono::high_resolution_clock::now();
        }
        if (abs(avg_loss - last_loss) < 1e-4) stagnate_count++; else stagnate_count = 0;
        last_loss = avg_loss;
        if (stagnate_count >= 4) { H += 2; stagnate_count = 0; }
        if (avg_loss < 0.05 && t % 20 == 0) for(int i=0; i<128 && next_idx < indices.size(); ++i) current_subset.push_back(indices[next_idx++]);
    }
    cublasDestroy(handle); return 0;
}
