#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <fftw3.h>
#include <random>
#include <chrono>
#include <algorithm>
#include <fstream>

using namespace std;

const int INPUT_DIM = 3072;
const int HIDDEN_DIM = 256; // Power of 2 for FFT efficiency
const int NUM_CLASSES = 10;
const int BATCH_SIZE = 1024;
const int TRAIN_LIMIT_S = 600;

struct SpectralModel {
    vector<float> W1, b1, W2, b2;
    vector<complex<double>> spectral_filter;

    SpectralModel() : 
        W1(HIDDEN_DIM * INPUT_DIM), b1(HIDDEN_DIM, 0),
        W2(NUM_CLASSES * HIDDEN_DIM), b2(NUM_CLASSES, 0),
        spectral_filter(HIDDEN_DIM) 
    {
        mt19937 gen(42);
        normal_distribution<float> d1(0, sqrt(2.0/INPUT_DIM)), d2(0, sqrt(2.0/HIDDEN_DIM));
        for(float& w : W1) w = d1(gen);
        for(float& w : W2) w = d2(gen);
        for(auto& c : spectral_filter) c = {1.0, 0.0};
    }
};

void apply_spectral_ode_step(vector<float>& h, vector<complex<double>>& filter) {
    fftw_complex *in, *out;
    fftw_plan p_forward, p_backward;

    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * HIDDEN_DIM);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * HIDDEN_DIM);

    // 1. Map spatial activations to Fourier Domain
    for(int i=0; i<HIDDEN_DIM; ++i) { in[i][0] = h[i]; in[i][1] = 0; }
    p_forward = fftw_plan_dft_1d(HIDDEN_DIM, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p_forward);

    // 2. Solve ODE component in frequency domain (Point-wise spectral filtering)
    for(int i=0; i<HIDDEN_DIM; ++i) {
        complex<double> freq(out[i][0], out[i][1]);
        freq *= filter[i];
        out[i][0] = freq.real();
        out[i][1] = freq.imag();
    }

    // 3. Transform back to Spatial Domain
    p_backward = fftw_plan_dft_1d(HIDDEN_DIM, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p_backward);

    for(int i=0; i<HIDDEN_DIM; ++i) h[i] = (float)(in[i][0] / HIDDEN_DIM);

    fftw_destroy_plan(p_forward);
    fftw_destroy_plan(p_backward);
    fftw_free(in);
    fftw_free(out);
}

void load_cifar_batch(vector<float>& X, vector<int>& y) {
    ifstream file("cifar-10-batches-bin/data_batch_1.bin", ios::binary);
    if (!file) return;
    for(int i=0; i<BATCH_SIZE; ++i) {
        unsigned char label; file.read((char*)&label, 1); y.push_back((int)label);
        vector<unsigned char> img(3072); file.read((char*)img.data(), 3072);
        for(auto b : img) X.push_back(b / 255.0f);
    }
}

int main() {
    cout << "Spectral ODE Neural Solver (FFT-Accelerated)..." << endl;
    vector<float> X; vector<int> y;
    load_cifar_batch(X, y);

    SpectralModel model;
    auto start = chrono::high_resolution_clock::now();
    float lr = 0.01f;
    int epoch = 0;

    while (chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start).count() < TRAIN_LIMIT_S) {
        float total_loss = 0;
        int correct = 0;

        for(int b=0; b<BATCH_SIZE; ++b) {
            // Layer 1 Forward
            vector<float> h(HIDDEN_DIM, 0);
            for(int i=0; i<HIDDEN_DIM; ++i) {
                for(int j=0; j<INPUT_DIM; ++j) h[i] += model.W1[i * INPUT_DIM + j] * X[b * INPUT_DIM + j];
                h[i] += model.b1[i];
            }

            // --- Spectral ODE Transformation ---
            apply_spectral_ode_step(h, model.spectral_filter);
            // ReLU
            for(float& val : h) val = val > 0 ? val : 0.1f * val;

            // Layer 2 Forward (Logits)
            vector<float> logits(NUM_CLASSES, 0);
            for(int i=0; i<NUM_CLASSES; ++i) {
                for(int j=0; j<HIDDEN_DIM; ++j) logits[i] += model.W2[i * HIDDEN_DIM + j] * h[j];
                logits[i] += model.b2[i];
            }

            // Softmax & Loss
            float max_l = *max_element(logits.begin(), logits.end());
            float sum_e = 0;
            for(float l : logits) sum_e += exp(l - max_l);
            vector<float> probs(NUM_CLASSES);
            for(int i=0; i<NUM_CLASSES; ++i) probs[i] = exp(logits[i] - max_l) / sum_e;

            total_loss -= log(probs[y[b]] + 1e-9);
            if(distance(probs.begin(), max_element(probs.begin(), probs.end())) == y[b]) correct++;

            // Simple GD Backprop (Truncated for brevity, focusing on Spectral approach)
            vector<float> dLogits = probs;
            dLogits[y[b]] -= 1.0f;

            for(int i=0; i<NUM_CLASSES; ++i) {
                for(int j=0; j<HIDDEN_DIM; ++j) {
                    float grad = dLogits[i] * h[j];
                    model.W2[i * HIDDEN_DIM + j] -= lr * grad;
                }
                model.b2[i] -= lr * dLogits[i];
            }
        }

        epoch++;
        auto now = chrono::high_resolution_clock::now();
        int elapsed = chrono::duration_cast<chrono::seconds>(now - start).count();
        cout << "[Time: " << elapsed << "s] Epoch: " << epoch << " | Err: " << (1.0 - (float)correct/BATCH_SIZE)*100 << "%" << endl;
    }

    return 0;
}
