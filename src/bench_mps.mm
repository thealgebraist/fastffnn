#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <iomanip>

using namespace std;

struct Stats {
    double median;
};

Stats compute_stats(vector<double>& times) {
    if (times.empty()) return {0};
    sort(times.begin(), times.end());
    return {times[times.size() / 2]};
}

void benchmark_mps(int M, int N, int K, int iterations = 100) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        cout << "MPS not supported" << endl;
        return;
    }
    
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    
    // Create buffers
    NSUInteger sizeA = M * K * sizeof(float);
    NSUInteger sizeB = K * N * sizeof(float);
    NSUInteger sizeC = M * N * sizeof(float);
    
    id<MTLBuffer> bufferA = [device newBufferWithLength:sizeA options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferB = [device newBufferWithLength:sizeB options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufferC = [device newBufferWithLength:sizeC options:MTLResourceStorageModeShared];
    
    // Initialize with 1.0f
    float* ptrA = (float*)bufferA.contents;
    for(int i=0; i<M*K; ++i) ptrA[i] = 1.0f;
    float* ptrB = (float*)bufferB.contents;
    for(int i=0; i<K*N; ++i) ptrB[i] = 1.0f;
    
    MPSMatrixDescriptor *descA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:K * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descB = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *descC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N * sizeof(float) dataType:MPSDataTypeFloat32];
    
    MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
    MPSMatrix *matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
    MPSMatrix *matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];
    
    MPSMatrixMultiplication *matmul = [[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:NO resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];
    
    // Warm up
    for(int i=0; i<10; ++i) {
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        [matmul encodeToCommandBuffer:commandBuffer leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
    
    vector<double> mps_times;
    for (int i = 0; i < iterations; ++i) {
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        auto start = chrono::high_resolution_clock::now();
        [matmul encodeToCommandBuffer:commandBuffer leftMatrix:matrixA rightMatrix:matrixB resultMatrix:matrixC];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        auto end = chrono::high_resolution_clock::now();
        mps_times.push_back(chrono::duration<double>(end - start).count());
    }
    
    double flops = 2.0 * M * N * K;
    Stats s_mps = compute_stats(mps_times);
    
    cout << "|" << setw(5) << M << "x" << setw(5) << N << "x" << setw(5) << K 
         << " | " << fixed << setprecision(2) << setw(10) << (flops / s_mps.median) / 1e9 << " GFLOPS"
         << " | " << setw(8) << s_mps.median * 1e6 << " us" << " | MPS" << endl;
}

int main() {
    cout << "Benchmarking Matrix Multiplication (MPS) on Apple Silicon" << endl;
    cout << "------------------------------------------------------------" << endl;
    
    vector<int> sizes = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    for (int s : sizes) {
        benchmark_mps(s, s, s, 50);
    }
    
    return 0;
}
