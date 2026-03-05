#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <Accelerate/Accelerate.h>

using namespace std;

struct Stats {
    double median;
};

Stats compute_stats(vector<double>& times) {
    if (times.empty()) return {0};
    sort(times.begin(), times.end());
    return {times[times.size() / 2]};
}

void gemm_accelerate(const float* A, const float* B, float* C, int M, int N, int K) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}

void benchmark(int M, int N, int K, int iterations = 100) {
    vector<float> A(M * K, 1.0f);
    vector<float> B(K * N, 1.0f);
    vector<float> C(M * N, 0.0f);
    
    for(int i=0; i<10; ++i) gemm_accelerate(A.data(), B.data(), C.data(), M, N, K);
    
    vector<double> times;
    for (int i = 0; i < iterations; ++i) {
        auto start = chrono::high_resolution_clock::now();
        gemm_accelerate(A.data(), B.data(), C.data(), M, N, K);
        auto end = chrono::high_resolution_clock::now();
        times.push_back(chrono::duration<double>(end - start).count());
    }
    
    double flops = 2.0 * M * N * K;
    Stats s = compute_stats(times);
    
    cout << "|" << setw(5) << M << "x" << setw(5) << N << "x" << setw(5) << K 
         << " | " << fixed << setprecision(2) << setw(10) << (flops / s.median) / 1e9 << " GFLOPS"
         << " | " << setw(8) << s.median * 1e6 << " us" << " |" << endl;
}

int main() {
    cout << "Tiling Sensitivity Analysis (Accelerate/AMX)" << endl;
    cout << "| Dimensions      | Accelerate GFLOPS | Time (us) |" << endl;
    cout << "|-----------------|-------------------|-----------|" << endl;
    
    vector<int> bases = {64, 128, 256, 512};
    for (int b : bases) {
        benchmark(b - 1, b - 1, b - 1, 200);
        benchmark(b, b, b, 200);
        benchmark(b + 1, b + 1, b + 1, 200);
        cout << "|-----------------|-------------------|-----------|" << endl;
    }
    
    return 0;
}
