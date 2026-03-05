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
const int TRAIN_LIMIT_S = 120;
const int NUM_TRAIN = 50000;
const int BATCH_SIZE = 256;

// A probability measure representation of an image using spatial and frequency moments
struct ImageMeasure {
    vector<float> moments;
};

// Compute the moments for the probability measure
ImageMeasure compute_measure(const float* img) {
    ImageMeasure m;
    m.moments.reserve(256);
    
    for (int c = 0; c < CHANNELS; ++c) {
        float sum = 0, sum_x = 0, sum_y = 0;
        float sum_xx = 0, sum_yy = 0, sum_xy = 0;
        
        for (int y = 0; y < IMG_SIZE; ++y) {
            for (int x = 0; x < IMG_SIZE; ++x) {
                float val = img[c * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x];
                sum += val;
                sum_x += val * x;
                sum_y += val * y;
                sum_xx += val * x * x;
                sum_yy += val * y * y;
                sum_xy += val * x * y;
            }
        }
        
        // Normalize to create a valid probability measure
        float inv_sum = sum > 0 ? 1.0f / sum : 0;
        float mu_x = sum_x * inv_sum;
        float mu_y = sum_y * inv_sum;
        float var_x = (sum_xx * inv_sum) - (mu_x * mu_x);
        float var_y = (sum_yy * inv_sum) - (mu_y * mu_y);
        float cov_xy = (sum_xy * inv_sum) - (mu_x * mu_y);
        
        m.moments.push_back(sum);
        m.moments.push_back(mu_x);
        m.moments.push_back(mu_y);
        m.moments.push_back(var_x);
        m.moments.push_back(var_y);
        m.moments.push_back(cov_xy);
        
        // Add localized grid measures (4x4 grid = 16 regions)
        int grid_size = 4;
        int step = IMG_SIZE / grid_size;
        for (int gy = 0; gy < grid_size; ++gy) {
            for (int gx = 0; gx < grid_size; ++gx) {
                float local_sum = 0;
                for (int y = gy * step; y < (gy + 1) * step; ++y) {
                    for (int x = gx * step; x < (gx + 1) * step; ++x) {
                        local_sum += img[c * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x];
                    }
                }
                m.moments.push_back(local_sum * inv_sum);
            }
        }
    }
    return m;
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

void transpose_matrix(const float* A, float* AT, int R, int C) {
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            AT[j * R + i] = A[i * C + j];
        }
    }
}

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

    cout << "Computing measure moments for all images..." << endl;
    vector<vector<float>> all_measures(NUM_TRAIN);
    int num_features = 0;
    
    for (int i = 0; i < NUM_TRAIN; ++i) {
        ImageMeasure m = compute_measure(&train_images[i * 3072]);
        all_measures[i] = m.moments;
        num_features = m.moments.size();
    }
    
    cout << "Measure representation features: " << num_features << endl;
    
    // Non-linear expansion to approximate deep metric spaces
    int expanded_features = num_features * 4;
    vector<float> W(expanded_features * NUM_CLASSES);
    vector<float> b(NUM_CLASSES, 0.0f);
    
    // Projection matrix to map features to expanded space
    vector<float> P(num_features * expanded_features);
    
    mt19937 gen(42);
    normal_distribution<float> dist(0.0f, sqrt(2.0f / expanded_features));
    for (auto& w : W) w = dist(gen);
    normal_distribution<float> dist_p(0.0f, sqrt(2.0f / num_features));
    for (auto& p : P) p = dist_p(gen);

    Adam adam_W(expanded_features * NUM_CLASSES);
    Adam adam_b(NUM_CLASSES);
    Adam adam_P(num_features * expanded_features);

    auto start_time = chrono::high_resolution_clock::now();
    int iterations = 0;
    
    vector<float> batch_features(BATCH_SIZE * num_features);
    vector<float> batch_expanded(BATCH_SIZE * expanded_features);
    vector<float> batch_logits(BATCH_SIZE * NUM_CLASSES);
    vector<float> dLogits(BATCH_SIZE * NUM_CLASSES);
    vector<float> dW(expanded_features * NUM_CLASSES);
    vector<float> dP(num_features * expanded_features);
    vector<float> dExpanded(BATCH_SIZE * expanded_features);
    vector<float> db(NUM_CLASSES);
    vector<uint8_t> batch_labels(BATCH_SIZE);

    vector<float> batch_expanded_T(expanded_features * BATCH_SIZE);
    vector<float> batch_features_T(num_features * BATCH_SIZE);
    vector<float> W_T(NUM_CLASSES * expanded_features);

    float rolling_loss = 0;
    int rolling_correct = 0;

    cout << "Training custom measure distribution test..." << endl;

    while (true) {
        auto now = chrono::high_resolution_clock::now();
        if (chrono::duration_cast<chrono::seconds>(now - start_time).count() >= TRAIN_LIMIT_S) break;

        for (int i = 0; i < BATCH_SIZE; ++i) {
            int img_idx = gen() % NUM_TRAIN;
            batch_labels[i] = train_labels[img_idx];
            for (int f = 0; f < num_features; ++f) {
                batch_features[i * num_features + f] = all_measures[img_idx][f];
            }
        }

        // P projection: batch_expanded (B x E) = batch_features (B x F) * P (F x E)
        vDSP_mmul(batch_features.data(), 1, P.data(), 1, batch_expanded.data(), 1, BATCH_SIZE, expanded_features, num_features);
        
        // ReLU activation
        for (int i = 0; i < BATCH_SIZE * expanded_features; ++i) {
            if (batch_expanded[i] < 0) batch_expanded[i] = 0;
        }

        // Logits
        vDSP_mmul(batch_expanded.data(), 1, W.data(), 1, batch_logits.data(), 1, BATCH_SIZE, NUM_CLASSES, expanded_features);

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

        // dW
        transpose_matrix(batch_expanded.data(), batch_expanded_T.data(), BATCH_SIZE, expanded_features);
        vDSP_mmul(batch_expanded_T.data(), 1, dLogits.data(), 1, dW.data(), 1, expanded_features, NUM_CLASSES, BATCH_SIZE);

        // Backprop to dExpanded
        transpose_matrix(W.data(), W_T.data(), expanded_features, NUM_CLASSES);
        vDSP_mmul(dLogits.data(), 1, W_T.data(), 1, dExpanded.data(), 1, BATCH_SIZE, expanded_features, NUM_CLASSES);
        
        // ReLU derivative
        for (int i = 0; i < BATCH_SIZE * expanded_features; ++i) {
            if (batch_expanded[i] <= 0) dExpanded[i] = 0;
        }

        // dP
        transpose_matrix(batch_features.data(), batch_features_T.data(), BATCH_SIZE, num_features);
        vDSP_mmul(batch_features_T.data(), 1, dExpanded.data(), 1, dP.data(), 1, num_features, expanded_features, BATCH_SIZE);

        float inv_B = 1.0f / BATCH_SIZE;
        vDSP_vsmul(dW.data(), 1, &inv_B, dW.data(), 1, expanded_features * NUM_CLASSES);
        vDSP_vsmul(dP.data(), 1, &inv_B, dP.data(), 1, num_features * expanded_features);
        vDSP_vsmul(db.data(), 1, &inv_B, db.data(), 1, NUM_CLASSES);

        adam_W.update(W, dW);
        adam_P.update(P, dP);
        adam_b.update(b, db);

        iterations++;
        if (iterations % 500 == 0) {
            cout << "Iter " << iterations << " | Loss: " << rolling_loss 
                 << " | Acc: " << (rolling_correct / (256.0f * 500)) * 100.0f << "%" << endl;
            rolling_correct = 0;
        }
    }
    
    cout << "Finished training." << endl;
    return 0;
}
