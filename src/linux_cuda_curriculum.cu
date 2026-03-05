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
template <class T, class U> bool operator==(const CudaManagedAllocator<T>&, const CudaManagedAllocator<U>&) { return true; }
template <class T, class U> bool operator!=(const CudaManagedAllocator<T>&, const CudaManagedAllocator<U>&) { return false; }

using CudaVector = std::vector<float, CudaManagedAllocator<float>>;

const int IMG_SIZE = 32;
const int CHANNELS = 3;
const int INPUT_DIM = IMG_SIZE * IMG_SIZE * CHANNELS;
const int MAX_NEURONS = 4096;
const int NUM_CLASSES = 10;
const int TRAIN_LIMIT_S = 600; 
const int INITIAL_IMAGES = 8192;
const int INITIAL_NEURONS = 64;
const int BATCH_SIZE = 64;

void download_cifar10() {
    struct stat buffer;
    if (stat("cifar-10-batches-bin/data_batch_1.bin", &buffer) != 0) {
        cout << "Downloading CIFAR-10 dataset..." << endl;
        int res = system("wget -qO- https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz | tar xz");
        if (res != 0) {
            cerr << "Failed to download CIFAR-10" << endl;
            exit(1);
        }
    }
}

bool load_cifar(const string& path, CudaVector& images, vector<uint8_t>& labels) {
    ifstream file(path, ios::binary);
    if (!file) return false;
    for (int i = 0; i < 10000; ++i) {
        uint8_t label; file.read((char*)&label, 1);
        labels.push_back(label);
        vector<uint8_t> raw(3072); file.read((char*)raw.data(), 3072);
        for(int p=0; p<3072; ++p) images.push_back(raw[p] / 255.0f);
    }
    return true;
}

void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int M, int N, int K) {
    float alpha = 1.0f;
    float beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, N,
                A, K,
                &beta,
                C, N));
}

void mmul_add(cublasHandle_t handle, const float* A, const float* B, float* C, int M, int N, int K) {
    float alpha = 1.0f;
    float beta = 1.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, N,
                A, K,
                &beta,
                C, N));
}

__global__ void scale_kernel(float* vec, float scalar, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) vec[i] *= scalar;
}

void scale_vector(CudaVector& vec, float scalar, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    scale_kernel<<<blocks, threads>>>(vec.data(), scalar, size);
    CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void radam_kernel(float* w, float* m, float* v, const float* g, 
                             float b1, float b2, float eps, float lr, 
                             float b1_t, float b2_t, float rho_inf, float rho_t, 
                             int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        m[i] = b1 * m[i] + (1.0f - b1) * g[i];
        v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
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

void radam_update(CudaVector& w, CudaVector& m, CudaVector& v, const CudaVector& g, int t, float lr, int size) {
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float rho_inf = 2.0f / (1.0f - b2) - 1.0f;
    float b1_t = powf(b1, t);
    float b2_t = powf(b2, t);
    float rho_t = rho_inf - 2.0f * t * b2_t / (1.0f - b2_t);
    
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    radam_kernel<<<blocks, threads>>>(w.data(), m.data(), v.data(), g.data(), 
                                      b1, b2, eps, lr, b1_t, b2_t, rho_inf, rho_t, size);
    CHECK_CUDA(cudaDeviceSynchronize());
}

float l2_dist_sq(const float* a, const float* b) {
    float sum = 0;
    for(int i=0; i<3072; ++i) { float d = a[i] - b[i]; sum += d*d; }
    return sum;
}

struct DynamicNetwork {
    int H; 
    CudaVector W1, b1, W2, b2;
    CudaVector bn_gamma, bn_beta;
    CudaVector mW1, vW1, mb1, vb1, mW2, vW2, mb2, vb2;
    CudaVector mG, vG, mB, vB;
    int t = 0;
    
    DynamicNetwork(int initial_h) : H(initial_h) {
        W1.resize(MAX_NEURONS * INPUT_DIM, 0);
        b1.resize(MAX_NEURONS, 0);
        W2.resize(NUM_CLASSES * MAX_NEURONS, 0);
        b2.resize(NUM_CLASSES, 0);
        bn_gamma.resize(MAX_NEURONS, 1.0f);
        bn_beta.resize(MAX_NEURONS, 0.0f);
        
        mW1.resize(MAX_NEURONS * INPUT_DIM, 0); vW1.resize(MAX_NEURONS * INPUT_DIM, 0);
        mb1.resize(MAX_NEURONS, 0); vb1.resize(MAX_NEURONS, 0);
        mW2.resize(NUM_CLASSES * MAX_NEURONS, 0); vW2.resize(NUM_CLASSES * MAX_NEURONS, 0);
        mb2.resize(NUM_CLASSES, 0); vb2.resize(NUM_CLASSES, 0);
        mG.resize(MAX_NEURONS, 0); vG.resize(MAX_NEURONS, 0);
        mB.resize(MAX_NEURONS, 0); vB.resize(MAX_NEURONS, 0);
        
        init_neurons(0, H);
    }
    
    void init_neurons(int start, int end) {
        mt19937 gen(42 + start);
        float s1 = sqrtf(2.0f / INPUT_DIM);
        float s2 = 1.0f / end; 
        normal_distribution<float> d1(0, s1), d2(0, s2);
        for(int i = start; i < end; ++i) {
            for(int j = 0; j < INPUT_DIM; ++j) W1[i * INPUT_DIM + j] = d1(gen);
            for(int j = 0; j < NUM_CLASSES; ++j) W2[j * MAX_NEURONS + i] = d2(gen);
        }
    }
    
    void add_neurons(int count) {
        if (H + count > MAX_NEURONS) return;
        init_neurons(H, H + count);
        H += count;
        cout << "Dynamic Expansion: Added " << count << " neurons. Width: " << H << endl;
    }
};

int main() {
    cout << "CUDA Intensive Curriculum Training..." << endl;
    download_cifar10();
    CudaVector all_images;
    vector<uint8_t> all_labels;
    for(int i=1; i<=5; ++i) load_cifar("cifar-10-batches-bin/data_batch_" + to_string(i) + ".bin", all_images, all_labels);
    
    DynamicNetwork net(INITIAL_NEURONS);
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    
    mt19937 gen(42);
    vector<int> indices(all_labels.size());
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), gen);
    
    vector<int> current_subset;
    current_subset.push_back(indices[0]);
    for(int i=1; i<INITIAL_IMAGES; ++i) {
        int best_cand = -1; float max_dist = -1;
        for(int j=0; j<10; ++j) {
            int cand = indices[INITIAL_IMAGES + (gen() % (indices.size() - INITIAL_IMAGES))];
            float d = l2_dist_sq(&all_images[current_subset.back() * 3072], &all_images[cand * 3072]);
            if(d > max_dist) { max_dist = d; best_cand = cand; }
        }
        current_subset.push_back(best_cand);
    }
    int next_idx = INITIAL_IMAGES;
    
    auto start_time = chrono::high_resolution_clock::now();
    auto last_print_time = start_time;
    float lr = 0.001f;
    float last_loss = 1e10;
    int stagnate_count = 0;
    
    CudaVector dW1(MAX_NEURONS * INPUT_DIM, 0), db1(MAX_NEURONS, 0);
    CudaVector dW2(NUM_CLASSES * MAX_NEURONS, 0), db2(NUM_CLASSES, 0);
    CudaVector dG(MAX_NEURONS, 0), dB(MAX_NEURONS, 0);
    CudaVector hs_buf(BATCH_SIZE * MAX_NEURONS, 0);
    CudaVector hs_norm_buf(BATCH_SIZE * MAX_NEURONS, 0);
    CudaVector dh_scaled(MAX_NEURONS, 0);
    CudaVector logits(NUM_CLASSES, 0);
    CudaVector dLogits(NUM_CLASSES, 0);
    
    while (true) {
        auto now = chrono::high_resolution_clock::now();
        double elapsed_s = chrono::duration_cast<chrono::seconds>(now - start_time).count();
        if (elapsed_s >= TRAIN_LIMIT_S) break;
        
        int correct = 0;
        int N_sub = current_subset.size();
        float total_loss = 0;
        
        fill(dW1.begin(), dW1.end(), 0.0f); fill(db1.begin(), db1.end(), 0.0f);
        fill(dW2.begin(), dW2.end(), 0.0f); fill(db2.begin(), db2.end(), 0.0f);
        fill(dG.begin(), dG.end(), 0.0f);   fill(dB.begin(), dB.end(), 0.0f);
        
        vector<int> batch_indices;
        for(int i=0; i<BATCH_SIZE; ++i) batch_indices.push_back(current_subset[gen() % N_sub]);
        
        vector<float> mu(net.H, 0), var(net.H, 0);
        for (int b = 0; b < BATCH_SIZE; ++b) {
            mmul(handle, net.W1.data(), &all_images[batch_indices[b] * 3072], &hs_buf[b * net.H], net.H, 1, INPUT_DIM);
            CHECK_CUDA(cudaDeviceSynchronize());
            for(int i=0; i<net.H; ++i) { hs_buf[b * net.H + i] += net.b1[i]; mu[i] += hs_buf[b * net.H + i]; }
        }
        for(int i=0; i<net.H; ++i) mu[i] /= BATCH_SIZE;
        for (int b = 0; b < BATCH_SIZE; ++b) {
            for(int i=0; i<net.H; ++i) {
                float d = hs_buf[b * net.H + i] - mu[i];
                var[i] += d * d;
            }
        }
        for(int i=0; i<net.H; ++i) var[i] = sqrtf(var[i] / BATCH_SIZE + 1e-5f);
        
        for (int b = 0; b < BATCH_SIZE; ++b) {
            int label = all_labels[batch_indices[b]];
            for(int i=0; i<net.H; ++i) {
                hs_norm_buf[b * net.H + i] = (hs_buf[b * net.H + i] - mu[i]) / var[i];
                float act = hs_norm_buf[b * net.H + i] * net.bn_gamma[i] + net.bn_beta[i];
                hs_buf[b * net.H + i] = act > 0 ? act : 0;
            }
            mmul(handle, net.W2.data(), &hs_buf[b * net.H], logits.data(), NUM_CLASSES, 1, net.H);
            CHECK_CUDA(cudaDeviceSynchronize());
            float max_l = -1e10; 
            for(int c=0; c<NUM_CLASSES; ++c) {
                logits[c] += net.b2[c]; if(logits[c] > max_l) max_l = logits[c];
            }
            float sum_e = 0; 
            for(int c=0; c<NUM_CLASSES; ++c) { logits[c] = expf(logits[c] - max_l); sum_e += logits[c]; }
            int pred = 0;
            for(int c=0; c<NUM_CLASSES; ++c) {
                logits[c] /= sum_e; if(logits[c] > logits[pred]) pred = c;
            }
            if(pred == label) correct++;
            total_loss -= logf(logits[label] + 1e-9f);
            
            for(int c=0; c<NUM_CLASSES; ++c) {
                dLogits[c] = logits[c] - (c == label ? 1.0f : 0.0f);
                db2[c] += dLogits[c];
            }
            mmul_add(handle, dLogits.data(), &hs_buf[b * net.H], dW2.data(), NUM_CLASSES, net.H, 1);
            CHECK_CUDA(cudaDeviceSynchronize());
            
            fill(dh_scaled.begin(), dh_scaled.end(), 0.0f);
            for(int i=0; i<net.H; ++i) {
                if(hs_buf[b * net.H + i] > 0) {
                    float dReLU = 0;
                    for(int c=0; c<NUM_CLASSES; ++c) dReLU += dLogits[c] * net.W2[c * MAX_NEURONS + i];
                    dG[i] += dReLU * hs_norm_buf[b * net.H + i];
                    dB[i] += dReLU;
                    float dNorm = dReLU * net.bn_gamma[i];
                    dh_scaled[i] = dNorm / var[i];
                    db1[i] += dh_scaled[i];
                }
            }
            mmul_add(handle, dh_scaled.data(), &all_images[batch_indices[b] * 3072], dW1.data(), net.H, INPUT_DIM, 1);
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        
        float avg_loss = total_loss / BATCH_SIZE;
        if (abs(avg_loss - last_loss) < 1e-4) stagnate_count++; else stagnate_count = 0;
        last_loss = avg_loss;
        
        if (chrono::duration_cast<chrono::seconds>(now - last_print_time).count() >= 1) {
            float err_pct = (1.0f - (correct / (float)BATCH_SIZE)) * 100.0f;
            cout << "[Time: " << (int)elapsed_s << "s] Err: " << err_pct << "% | Loss: " << avg_loss << " | Subset: " << N_sub << " | Neurons: " << net.H << endl;
            last_print_time = now;
        }

        if (stagnate_count >= 4) { net.add_neurons(2); stagnate_count = 0; }
        
        net.t++;
        float inv_B = 1.0f / BATCH_SIZE;
        scale_vector(dW1, inv_B, net.H * INPUT_DIM);
        scale_vector(dW2, inv_B, NUM_CLASSES * MAX_NEURONS);
        scale_vector(dG, inv_B, net.H);
        scale_vector(dB, inv_B, net.H);
        for(int i=0; i<net.H; ++i) db1[i] *= inv_B;
        for(int i=0; i<NUM_CLASSES; ++i) db2[i] *= inv_B;

        int dummy_t = net.t;
        radam_update(net.W1, net.mW1, net.vW1, dW1, dummy_t, lr, net.H * INPUT_DIM);
        dummy_t = net.t;
        radam_update(net.b1, net.mb1, net.vb1, db1, dummy_t, lr, net.H);
        dummy_t = net.t;
        radam_update(net.W2, net.mW2, net.vW2, dW2, dummy_t, lr, NUM_CLASSES * MAX_NEURONS);
        dummy_t = net.t;
        radam_update(net.b2, net.mb2, net.vb2, db2, dummy_t, lr, NUM_CLASSES);
        dummy_t = net.t;
        radam_update(net.bn_gamma, net.mG, net.vG, dG, dummy_t, lr, net.H);
        dummy_t = net.t;
        radam_update(net.bn_beta, net.mB, net.vB, dB, dummy_t, lr, net.H);
        
        if (avg_loss < 0.05) {
            if (net.t % 50 == 0) {
                cout << "[Curriculum] Adding 128 more images." << endl;
                for(int i=0; i<128 && next_idx < indices.size(); ++i) current_subset.push_back(indices[next_idx++]);
            }
        }
    }
    cublasDestroy(handle);
    return 0;
}
