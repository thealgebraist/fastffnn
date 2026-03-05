#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <Accelerate/Accelerate.h>

using namespace std;

const int IMG_SIZE = 32;
const int CHANNELS = 3;
const int NUM_CLASSES = 10;
const int MAX_K = 16;
const int TRAIN_LIMIT_S = 120;
const int BATCH_SIZE = 128;

// Create 1D DCT-II matrix
vector<float> make_dct_matrix(int N) {
    vector<float> C(N * N);
    for (int u = 0; u < N; ++u) {
        float alpha = (u == 0) ? sqrt(1.0f / N) : sqrt(2.0f / N);
        for (int x = 0; x < N; ++x) {
            C[u * N + x] = alpha * cos(M_PI * u * (2 * x + 1) / (2.0f * N));
        }
    }
    return C;
}

// 2D DCT
void dct2d_fast(const vector<float>& C, const float* X, float* Y, int N) {
    vector<float> CT(N * N);
    for (int i=0; i<N; ++i) for (int j=0; j<N; ++j) CT[j*N+i] = C[i*N+j];
    vector<float> CX(N * N, 0);
    for (int i=0; i<N; ++i)
        for (int k=0; k<N; ++k)
            for (int j=0; j<N; ++j)
                CX[i*N+j] += C[i*N+k] * X[k*N+j];
    for (int i=0; i<N; ++i) {
        for (int j=0; j<N; ++j) Y[i*N+j] = 0;
        for (int k=0; k<N; ++k)
            for (int j=0; j<N; ++j)
                Y[i*N+j] += CX[i*N+k] * CT[k*N+j];
    }
}

// Extract kxk low-frequencies and transform back to spatial kxk block
void idct2d_downsample_fast(const vector<float>& C_k, const float* Y, int N, float* X_k, int k) {
    vector<float> CkT(k * k);
    for (int i=0; i<k; ++i) for (int j=0; j<k; ++j) CkT[j*k+i] = C_k[i*k+j];
    
    vector<float> Y_k(k * k);
    for (int i=0; i<k; ++i) for (int j=0; j<k; ++j) Y_k[i*k+j] = Y[i*N+j];
    
    vector<float> CkT_Yk(k * k, 0);
    for (int i=0; i<k; ++i)
        for (int u=0; u<k; ++u)
            for (int j=0; j<k; ++j)
                CkT_Yk[i*k+j] += CkT[i*k+u] * Y_k[u*k+j];
                
    for (int i=0; i<k; ++i) {
        for (int j=0; j<k; ++j) X_k[i*k+j] = 0;
        for (int u=0; u<k; ++u)
            for (int j=0; j<k; ++j)
                X_k[i*k+j] += CkT_Yk[i*k+u] * C_k[u*k+j];
    }
}

void transpose_matrix(const float* A, float* AT, int R, int C) {
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            AT[j * R + i] = A[i * C + j];
        }
    }
}

bool load_cifar(const string& path, vector<float>& images, vector<uint8_t>& labels) {
    ifstream file(path, ios::binary);
    if (!file) return false;
    for (int i = 0; i < 10000; ++i) {
        uint8_t label;
        file.read((char*)&label, 1);
        labels.push_back(label);
        vector<uint8_t> img_raw(3072);
        file.read((char*)img_raw.data(), 3072);
        for (int p = 0; p < 3072; ++p) {
            images.push_back(img_raw[p] / 255.0f);
        }
    }
    return true;
}

struct Adam {
    vector<float> m, v;
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    float lr = 0.005f;
    int t = 0;
    Adam(int size) : m(size, 0), v(size, 0) {}
    void update(vector<float>& weights, const vector<float>& grads) {
        t++;
        float lr_t = lr * sqrt(1.0f - pow(beta2, t)) / (1.0f - pow(beta1, t));
        for (size_t i = 0; i < weights.size(); ++i) {
            m[i] = beta1 * m[i] + (1.0f - beta1) * grads[i];
            v[i] = beta2 * v[i] + (1.0f - beta2) * grads[i] * grads[i];
            weights[i] -= lr_t * m[i] / (sqrt(v[i]) + eps);
        }
    }
};

int main() {
    cout << "Loading CIFAR-10 data..." << endl;
    vector<float> train_images;
    vector<uint8_t> train_labels;
    for (int i = 1; i <= 5; ++i) {
        string path = "../amx/cifar-10-batches-bin/data_batch_" + to_string(i) + ".bin";
        if (!load_cifar(path, train_images, train_labels)) {
            cerr << "Failed to load " << path << endl;
            return 1;
        }
    }

    int total_features = 0;
    for (int k = 1; k <= 16; ++k) total_features += k * k;
    total_features *= 3;
    cout << "Total features per image (1x1 to 16x16, 3 channels): " << total_features << endl;

    vector<float> features(50000 * total_features);
    
    vector<vector<float>> C_k(17);
    for (int k = 1; k <= 16; ++k) C_k[k] = make_dct_matrix(k);
    vector<float> C_32 = make_dct_matrix(32);

    cout << "Precomputing Fourier downsampled features for all images..." << endl;
    for (int img_idx = 0; img_idx < 50000; ++img_idx) {
        int feat_offset = img_idx * total_features;
        int idx = 0;
        for (int c = 0; c < 3; ++c) {
            const float* X = &train_images[img_idx * 3072 + c * 1024];
            float Y[1024];
            dct2d_fast(C_32, X, Y, 32);
            for (int k = 1; k <= 16; ++k) {
                float X_k[256];
                idct2d_downsample_fast(C_k[k], Y, 32, X_k, k);
                for (int i = 0; i < k * k; ++i) {
                    features[feat_offset + idx++] = X_k[i];
                }
            }
        }
    }
    cout << "Feature extraction complete. Starting 120s optimization..." << endl;

    vector<float> W(total_features * NUM_CLASSES);
    vector<float> b(NUM_CLASSES, 0.0f);
    Adam adam_W(total_features * NUM_CLASSES);
    Adam adam_b(NUM_CLASSES);

    mt19937 gen(42);
    normal_distribution<float> dist(0.0f, sqrt(2.0f / total_features));
    for (auto& w : W) w = dist(gen);

    auto start_time = chrono::high_resolution_clock::now();
    int iterations = 0;
    
    vector<float> batch_features(BATCH_SIZE * total_features);
    vector<float> batch_features_T(total_features * BATCH_SIZE);
    vector<float> batch_logits(BATCH_SIZE * NUM_CLASSES);
    vector<float> dLogits(BATCH_SIZE * NUM_CLASSES);
    vector<float> dW(total_features * NUM_CLASSES);
    vector<float> db(NUM_CLASSES);
    vector<uint8_t> batch_labels(BATCH_SIZE);

    float rolling_loss = 0;
    int rolling_correct = 0;

    while (true) {
        auto now = chrono::high_resolution_clock::now();
        if (chrono::duration_cast<chrono::seconds>(now - start_time).count() >= TRAIN_LIMIT_S) break;

        for (int i = 0; i < BATCH_SIZE; ++i) {
            int img_idx = gen() % 50000;
            batch_labels[i] = train_labels[img_idx];
            for (int f = 0; f < total_features; ++f) {
                batch_features[i * total_features + f] = features[img_idx * total_features + f];
            }
        }

        // batch_logits (B x C) = batch_features (B x F) * W^T (F x C)
        // Since W is stored as C x F in standard layout? No, we store W as F x C
        // Matrix W is total_features rows, NUM_CLASSES cols
        vDSP_mmul(batch_features.data(), 1, W.data(), 1, batch_logits.data(), 1, BATCH_SIZE, NUM_CLASSES, total_features);

        for (int i = 0; i < NUM_CLASSES; ++i) db[i] = 0;
        
        float batch_loss = 0;
        int batch_correct = 0;
        
        for (int i = 0; i < BATCH_SIZE; ++i) {
            float max_l = -1e9f;
            for (int c = 0; c < NUM_CLASSES; ++c) {
                batch_logits[i * NUM_CLASSES + c] += b[c];
                if (batch_logits[i * NUM_CLASSES + c] > max_l) max_l = batch_logits[i * NUM_CLASSES + c];
            }
            float sum_e = 0;
            for (int c = 0; c < NUM_CLASSES; ++c) {
                batch_logits[i * NUM_CLASSES + c] = expf(batch_logits[i * NUM_CLASSES + c] - max_l);
                sum_e += batch_logits[i * NUM_CLASSES + c];
            }
            int label = batch_labels[i];
            int pred = 0;
            for (int c = 0; c < NUM_CLASSES; ++c) {
                batch_logits[i * NUM_CLASSES + c] /= sum_e;
                dLogits[i * NUM_CLASSES + c] = batch_logits[i * NUM_CLASSES + c] - (c == label ? 1.0f : 0.0f);
                db[c] += dLogits[i * NUM_CLASSES + c];
                if (batch_logits[i * NUM_CLASSES + c] > batch_logits[i * NUM_CLASSES + pred]) pred = c;
            }
            batch_loss -= logf(batch_logits[i * NUM_CLASSES + label] + 1e-9f);
            if (pred == label) batch_correct++;
        }
        
        if (iterations == 0) {
            rolling_loss = batch_loss / BATCH_SIZE;
        } else {
            rolling_loss = 0.99f * rolling_loss + 0.01f * (batch_loss / BATCH_SIZE);
        }
        rolling_correct += batch_correct;

        // dW (F x C) = batch_features_T (F x B) * dLogits (B x C)
        transpose_matrix(batch_features.data(), batch_features_T.data(), BATCH_SIZE, total_features);
        vDSP_mmul(batch_features_T.data(), 1, dLogits.data(), 1, dW.data(), 1, total_features, NUM_CLASSES, BATCH_SIZE);

        float inv_B = 1.0f / BATCH_SIZE;
        vDSP_vsmul(dW.data(), 1, &inv_B, dW.data(), 1, total_features * NUM_CLASSES);
        vDSP_vsmul(db.data(), 1, &inv_B, db.data(), 1, NUM_CLASSES);

        adam_W.update(W, dW);
        adam_b.update(b, db);

        iterations++;
        if (iterations % 1000 == 0) {
            cout << "Iter " << iterations << " | Loss: " << rolling_loss 
                 << " | Acc: " << (rolling_correct / (128.0f * 1000)) * 100.0f << "%" << endl;
            rolling_correct = 0;
        }
    }
    
    cout << "Finished training. Final updates complete." << endl;
    return 0;
}
