#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <Accelerate/Accelerate.h>
#include <arm_neon.h>

using namespace std;

// Statistical helper to compute mean and std dev
struct Stats {
    double mean;
    double stddev;
    double median;
};

Stats compute_stats(vector<double>& times) {
    if (times.empty()) return {0, 0, 0};
    sort(times.begin(), times.end());
    
    // Remove outliers (bottom 5% and top 5%)
    size_t start = times.size() * 0.05;
    size_t end = times.size() * 0.95;
    if (end <= start) {
        start = 0;
        end = times.size();
    }
    
    double sum = 0;
    for (size_t i = start; i < end; ++i) sum += times[i];
    double mean = sum / (end - start);
    
    double sq_sum = 0;
    for (size_t i = start; i < end; ++i) sq_sum += (times[i] - mean) * (times[i] - mean);
    double stddev = sqrt(sq_sum / (end - start));
    
    double median = times[times.size() / 2];
    
    return {mean, stddev, median};
}

// 1. NEON Implementation (Simple 4x4 tile)
void gemm_neon(const float* A, const float* B, float* C, int M, int N, int K) {
    // Basic implementation for benchmarking
    // Assume M, N, K are multiples of 4 for simplicity in this benchmark
    for (int i = 0; i < M; i += 4) {
        for (int j = 0; j < N; j += 4) {
            float32x4_t c0 = vdupq_n_f32(0);
            float32x4_t c1 = vdupq_n_f32(0);
            float32x4_t c2 = vdupq_n_f32(0);
            float32x4_t c3 = vdupq_n_f32(0);
            
            for (int k = 0; k < K; ++k) {
                float32x4_t b = vld1q_f32(B + k * N + j);
                c0 = vfmaq_n_f32(c0, b, A[i * K + k]);
                c1 = vfmaq_n_f32(c1, b, A[(i+1) * K + k]);
                c2 = vfmaq_n_f32(c2, b, A[(i+2) * K + k]);
                c3 = vfmaq_n_f32(c3, b, A[(i+3) * K + k]);
            }
            
            vst1q_f32(C + i * N + j, c0);
            vst1q_f32(C + (i+1) * N + j, c1);
            vst1q_f32(C + (i+2) * N + j, c2);
            vst1q_f32(C + (i+3) * N + j, c3);
        }
    }
}

// 2. Accelerate Implementation
void gemm_accelerate(const float* A, const float* B, float* C, int M, int N, int K) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}

void benchmark(int M, int N, int K, int iterations = 100) {
    vector<float> A(M * K, 1.0f);
    vector<float> B(K * N, 1.0f);
    vector<float> C(M * N, 0.0f);
    
    // Warm up
    for(int i=0; i<10; ++i) gemm_accelerate(A.data(), B.data(), C.data(), M, N, K);
    
    vector<double> acc_times;
    for (int i = 0; i < iterations; ++i) {
        auto start = chrono::high_resolution_clock::now();
        gemm_accelerate(A.data(), B.data(), C.data(), M, N, K);
        auto end = chrono::high_resolution_clock::now();
        acc_times.push_back(chrono::duration<double>(end - start).count());
    }
    
    vector<double> neon_times;
    // Only run NEON if dimensions are compatible with our simple kernel
    if (M % 4 == 0 && N % 4 == 0) {
        for (int i = 0; i < iterations; ++i) {
            auto start = chrono::high_resolution_clock::now();
            gemm_neon(A.data(), B.data(), C.data(), M, N, K);
            auto end = chrono::high_resolution_clock::now();
            neon_times.push_back(chrono::duration<double>(end - start).count());
        }
    }
    
    double flops = 2.0 * M * N * K;
    Stats s_acc = compute_stats(acc_times);
    
    cout << "|" << setw(5) << M << "x" << setw(5) << N << "x" << setw(5) << K 
         << " | " << fixed << setprecision(2) << setw(10) << (flops / s_acc.median) / 1e9 << " GFLOPS"
         << " | " << setw(8) << s_acc.median * 1e6 << " us" << " |";
    
    if (!neon_times.empty()) {
        Stats s_neon = compute_stats(neon_times);
        cout << fixed << setprecision(2) << setw(10) << (flops / s_neon.median) / 1e9 << " GFLOPS"
             << " | " << setw(8) << s_neon.median * 1e6 << " us" << " |";
    } else {
        cout << "    N/A     |    N/A    |";
    }
    cout << endl;
}

int main() {
    cout << "Benchmarking Matrix Multiplication (FP32) on Apple Silicon" << endl;
    cout << "------------------------------------------------------------" << endl;
    cout << "| Dimensions      | Accelerate GFLOPS | Time (us) | NEON GFLOPS | Time (us) |" << endl;
    cout << "|-----------------|-------------------|-----------|-------------|-----------|" << endl;
    
    vector<int> sizes = {16, 32, 64, 128, 256, 512, 1024};
    for (int s : sizes) {
        benchmark(s, s, s, 100);
    }
    
    cout << "------------------------------------------------------------" << endl;
    cout << "Testing Rectangular Matrices (Common in FFNNs)" << endl;
    benchmark(1, 1024, 1024, 1000); // Inference batch size 1
    benchmark(8, 1024, 1024, 500);
    benchmark(32, 1024, 1024, 200);
    benchmark(128, 1024, 1024, 100);
    
    return 0;
}
