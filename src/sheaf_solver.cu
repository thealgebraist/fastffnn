#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>
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

const int INPUT_DIM = 3072;
const int HIDDEN_DIM = 256;
const int NUM_CLASSES = 10;
const int BATCH_SIZE = 50000;
const int TRAIN_LIMIT_S = 600;

__global__ void matrix_sub_alpha_kernel(float* out, const float* A, const float* B, float alpha, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = A[i] - alpha * B[i];
    }
}

__global__ void lrelu_forward_kernel(float* out, const float* in, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = in[i] > 0 ? in[i] : 0.1f * in[i];
    }
}

__global__ void lrelu_backward_kernel(float* dPre, const float* dH1, const float* H_pre, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        dPre[i] = dH1[i] * (H_pre[i] > 0 ? 1.0f : 0.1f);
    }
}

__global__ void symmetrize_kernel(float* dL_sym, const float* dL, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * N) {
        int r = idx % N;
        int c = idx / N;
        dL_sym[r + c * N] = dL[r + c * N] + dL[c + r * N];
    }
}

__global__ void softmax_loss_kernel(const float* logits, const uint8_t* labels, float* dLogits, float* loss, int* correct, int B, int C) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B) {
        float max_l = -1e10;
        for(int c=0; c<C; ++c) { float l = logits[c + b * C]; if(l > max_l) max_l = l; }
        float sum_e = 0;
        for(int c=0; c<C; ++c) sum_e += expf(logits[c + b * C] - max_l);
        int label = labels[b];
        float prob = expf(logits[label + b * C] - max_l) / sum_e;
        atomicAdd(loss, -logf(prob + 1e-9f));
        int pred = 0; float max_p = -1.0;
        for(int c=0; c<C; ++c) {
            float p = expf(logits[c + b * C] - max_l) / sum_e;
            dLogits[c + b * C] = p - (c == label ? 1.0f : 0.0f);
            if(p > max_p) { max_p = p; pred = c; }
        }
        if(pred == label) atomicAdd(correct, 1);
    }
}

__global__ void update_weights_kernel(float* W, const float* grad, float lr, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) W[i] -= lr * grad[i];
}

void load_cifar(vector<float>& images, vector<uint8_t>& labels) {
    for (int b_idx = 1; b_idx <= 5; ++b_idx) {
        ifstream file("cifar-10-batches-bin/data_batch_" + to_string(b_idx) + ".bin", ios::binary);
        if(!file) continue;
        for (int i = 0; i < 10000; ++i) {
            uint8_t label; file.read((char*)&label, 1); labels.push_back(label);
            vector<uint8_t> raw(3072); file.read((char*)raw.data(), 3072);
            for(int p=0; p<3072; ++p) images.push_back(raw[p] / 255.0f);
        }
    }
}

int main() {
    cout << "Neural Sheaf Diffusion Solver (CUDA Optimized for H200)..." << endl;
    vector<float> cpu_images; vector<uint8_t> cpu_labels;
    load_cifar(cpu_images, cpu_labels);

    float *d_X; uint8_t *d_labels;
    CHECK_CUDA(cudaMalloc(&d_X, BATCH_SIZE * INPUT_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_labels, BATCH_SIZE * sizeof(uint8_t)));
    // X is 3072 x B in col-major
    CHECK_CUDA(cudaMemcpy(d_X, cpu_images.data(), BATCH_SIZE * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_labels, cpu_labels.data(), BATCH_SIZE * sizeof(uint8_t), cudaMemcpyHostToDevice));

    float *W_in, *W_out, *L_base;
    CHECK_CUDA(cudaMalloc(&W_in, HIDDEN_DIM * INPUT_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&W_out, NUM_CLASSES * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&L_base, HIDDEN_DIM * HIDDEN_DIM * sizeof(float)));
    
    mt19937 gen(42);
    normal_distribution<float> d1(0, sqrtf(2.0f/INPUT_DIM)), d2(0, sqrtf(2.0f/HIDDEN_DIM));
    vector<float> h_W_in(HIDDEN_DIM * INPUT_DIM), h_W_out(NUM_CLASSES * HIDDEN_DIM), h_L_base(HIDDEN_DIM * HIDDEN_DIM);
    for(float& w : h_W_in) w = d1(gen);
    for(float& w : h_W_out) w = d2(gen);
    for(float& w : h_L_base) w = d2(gen) * 0.1f; // Small initial Laplacian
    
    CHECK_CUDA(cudaMemcpy(W_in, h_W_in.data(), HIDDEN_DIM * INPUT_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(W_out, h_W_out.data(), NUM_CLASSES * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(L_base, h_L_base.data(), HIDDEN_DIM * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle; CHECK_CUBLAS(cublasCreate(&handle));

    float *d_H0, *d_L, *d_H_diff, *d_H_pre, *d_H1, *d_Logits;
    CHECK_CUDA(cudaMalloc(&d_H0, BATCH_SIZE * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_L, HIDDEN_DIM * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_H_diff, BATCH_SIZE * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_H_pre, BATCH_SIZE * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_H1, BATCH_SIZE * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Logits, BATCH_SIZE * NUM_CLASSES * sizeof(float)));

    float *d_dLogits, *d_dW_out, *d_dH1, *d_dPre, *d_dPre_L, *d_dH0, *d_dL, *d_dL_sym, *d_dL_base, *d_dW_in;
    CHECK_CUDA(cudaMalloc(&d_dLogits, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dW_out, NUM_CLASSES * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dH1, BATCH_SIZE * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dPre, BATCH_SIZE * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dPre_L, BATCH_SIZE * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dH0, BATCH_SIZE * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dL, HIDDEN_DIM * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dL_sym, HIDDEN_DIM * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dL_base, HIDDEN_DIM * HIDDEN_DIM * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dW_in, HIDDEN_DIM * INPUT_DIM * sizeof(float)));

    float* loss_gpu; int* correct_gpu;
    CHECK_CUDA(cudaMallocManaged(&loss_gpu, sizeof(float)));
    CHECK_CUDA(cudaMallocManaged(&correct_gpu, sizeof(int)));

    auto start_time = chrono::high_resolution_clock::now();
    float lr = 0.05f; 
    float alpha = 0.1f; // Diffusion step size
    int iter = 0;

    int num_blocks_B_H = (BATCH_SIZE * HIDDEN_DIM + 255) / 256;
    int num_blocks_B_C = (BATCH_SIZE * NUM_CLASSES + 255) / 256;
    int num_blocks_H_H = (HIDDEN_DIM * HIDDEN_DIM + 255) / 256;

    while (chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start_time).count() < TRAIN_LIMIT_S) {
        float one = 1.0f, zero = 0.0f;

        // 1. L = L_base^T L_base
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, &one, L_base, HIDDEN_DIM, L_base, HIDDEN_DIM, &zero, d_L, HIDDEN_DIM));

        // 2. H0 = W_in X  (W_in: 256x3072, X: 3072xB -> H0: 256xB)
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, HIDDEN_DIM, BATCH_SIZE, INPUT_DIM, &one, W_in, HIDDEN_DIM, d_X, INPUT_DIM, &zero, d_H0, HIDDEN_DIM));

        // 3. H_diff = L H0  (L: 256x256, H0: 256xB -> H_diff: 256xB)
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, HIDDEN_DIM, BATCH_SIZE, HIDDEN_DIM, &one, d_L, HIDDEN_DIM, d_H0, HIDDEN_DIM, &zero, d_H_diff, HIDDEN_DIM));

        // 4. H_pre = H0 - alpha * H_diff
        matrix_sub_alpha_kernel<<<num_blocks_B_H, 256>>>(d_H_pre, d_H0, d_H_diff, alpha, BATCH_SIZE * HIDDEN_DIM);

        // 5. H1 = LReLU(H_pre)
        lrelu_forward_kernel<<<num_blocks_B_H, 256>>>(d_H1, d_H_pre, BATCH_SIZE * HIDDEN_DIM);

        // 6. Logits = W_out H1 (W_out: 10x256, H1: 256xB -> Logits: 10xB)
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, NUM_CLASSES, BATCH_SIZE, HIDDEN_DIM, &one, W_out, NUM_CLASSES, d_H1, HIDDEN_DIM, &zero, d_Logits, NUM_CLASSES));

        *loss_gpu = 0; *correct_gpu = 0;
        softmax_loss_kernel<<<num_blocks_B_C, 256>>>(d_Logits, d_labels, d_dLogits, loss_gpu, correct_gpu, BATCH_SIZE, NUM_CLASSES);
        CHECK_CUDA(cudaDeviceSynchronize());

        // BACKWARD PASS
        float invB = 1.0f / BATCH_SIZE;

        // 7. dW_out = dLogits H1^T (dLogits: 10xB, H1^T: Bx256 -> dW_out: 10x256)
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, NUM_CLASSES, HIDDEN_DIM, BATCH_SIZE, &invB, d_dLogits, NUM_CLASSES, d_H1, HIDDEN_DIM, &zero, d_dW_out, NUM_CLASSES));

        // 8. dH1 = W_out^T dLogits (W_out^T: 256x10, dLogits: 10xB -> dH1: 256xB)
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, HIDDEN_DIM, BATCH_SIZE, NUM_CLASSES, &one, W_out, NUM_CLASSES, d_dLogits, NUM_CLASSES, &zero, d_dH1, HIDDEN_DIM));

        // 9. dPre = dH1 * LReLU'(H_pre)
        lrelu_backward_kernel<<<num_blocks_B_H, 256>>>(d_dPre, d_dH1, d_H_pre, BATCH_SIZE * HIDDEN_DIM);

        // 10. dPre_L = L dPre (L: 256x256, dPre: 256xB -> dPre_L: 256xB)
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, HIDDEN_DIM, BATCH_SIZE, HIDDEN_DIM, &one, d_L, HIDDEN_DIM, d_dPre, HIDDEN_DIM, &zero, d_dPre_L, HIDDEN_DIM));

        // 11. dH0 = dPre - alpha * dPre_L
        matrix_sub_alpha_kernel<<<num_blocks_B_H, 256>>>(d_dH0, d_dPre, d_dPre_L, alpha, BATCH_SIZE * HIDDEN_DIM);

        // 12. dL = dPre H0^T * (-alpha / B)
        float neg_alpha_invB = -alpha * invB;
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, HIDDEN_DIM, HIDDEN_DIM, BATCH_SIZE, &neg_alpha_invB, d_dPre, HIDDEN_DIM, d_H0, HIDDEN_DIM, &zero, d_dL, HIDDEN_DIM));

        // 13. dL_sym = dL + dL^T
        symmetrize_kernel<<<num_blocks_H_H, 256>>>(d_dL_sym, d_dL, HIDDEN_DIM);

        // 14. dL_base = L_base dL_sym
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, HIDDEN_DIM, HIDDEN_DIM, HIDDEN_DIM, &one, L_base, HIDDEN_DIM, d_dL_sym, HIDDEN_DIM, &zero, d_dL_base, HIDDEN_DIM));

        // 15. dW_in = dH0 X^T * (1/B)
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, HIDDEN_DIM, INPUT_DIM, BATCH_SIZE, &invB, d_dH0, HIDDEN_DIM, d_X, INPUT_DIM, &zero, d_dW_in, HIDDEN_DIM));

        // UPDATE WEIGHTS
        update_weights_kernel<<<(NUM_CLASSES * HIDDEN_DIM + 255)/256, 256>>>(W_out, d_dW_out, lr, NUM_CLASSES * HIDDEN_DIM);
        update_weights_kernel<<<num_blocks_H_H, 256>>>(L_base, d_dL_base, lr, HIDDEN_DIM * HIDDEN_DIM);
        update_weights_kernel<<<(HIDDEN_DIM * INPUT_DIM + 255)/256, 256>>>(W_in, d_dW_in, lr, HIDDEN_DIM * INPUT_DIM);

        iter++;
        int current_s = chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start_time).count();
        if (iter % 10 == 0) {
            cout << "[Time: " << current_s << "s] Iter: " << iter << " | Err: " << (1.0f - (*correct_gpu / (float)BATCH_SIZE)) * 100.0f << "% | Loss: " << *loss_gpu / BATCH_SIZE << endl;
        }
    }

    cublasDestroy(handle);
    return 0;
}
