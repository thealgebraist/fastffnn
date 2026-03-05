#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <Accelerate/Accelerate.h>

using namespace std;

const int NUM_EYES = 16;
const int IMG_SIZE = 32;
const int CHANNELS = 3;
const int INPUT_DIM = NUM_EYES * CHANNELS;
const int HIDDEN_DIM = 64;
const int NUM_CLASSES = 10;
const int TRAIN_LIMIT_S = 120;

struct Eye {
    float x, y, sigma;
    float dx, dy, dsigma;
    float vx, vy, vsigma; // Momentum for R-Adam
    float mx, my, msigma;
};

struct Network {
    vector<Eye> eyes;
    vector<float> W1, W2; // Weights
    vector<float> b1, b2; // Biases
    vector<float> mW1, vW1, mW2, vW2; // Adam states
};

void init_network(Network& net) {
    mt19937 gen(42);
    uniform_real_distribution<float> dist_pos(0.2f, 0.8f);
    uniform_real_distribution<float> dist_sigma(0.05f, 0.2f);
    
    for (int i = 0; i < NUM_EYES; ++i) {
        net.eyes.push_back({dist_pos(gen), dist_pos(gen), dist_sigma(gen), 0, 0, 0, 0, 0, 0, 0, 0, 0});
    }
    
    float scale1 = sqrtf(2.0f / INPUT_DIM);
    float scale2 = sqrtf(2.0f / HIDDEN_DIM);
    normal_distribution<float> d1(0, scale1);
    normal_distribution<float> d2(0, scale2);
    
    net.W1.resize(INPUT_DIM * HIDDEN_DIM);
    net.mW1.resize(INPUT_DIM * HIDDEN_DIM, 0);
    net.vW1.resize(INPUT_DIM * HIDDEN_DIM, 0);
    for (auto& w : net.W1) w = d1(gen);
    
    net.W2.resize(HIDDEN_DIM * NUM_CLASSES);
    net.mW2.resize(HIDDEN_DIM * NUM_CLASSES, 0);
    net.vW2.resize(HIDDEN_DIM * NUM_CLASSES, 0);
    for (auto& w : net.W2) w = d2(gen);
    
    net.b1.resize(HIDDEN_DIM, 0);
    net.b2.resize(NUM_CLASSES, 0);
}

// Gaussian sampling kernel
float sample_eye(const float* img, int c, float ex, float ey, float esigma) {
    float sum_val = 0;
    float sum_weight = 0;
    float s2 = 2.0f * esigma * esigma;
    
    for (int y = 0; y < IMG_SIZE; ++y) {
        float dy = (y / (float)IMG_SIZE) - ey;
        for (int x = 0; x < IMG_SIZE; ++x) {
            float dx = (x / (float)IMG_SIZE) - ex;
            float dist_sq = dx*dx + dy*dy;
            float weight = expf(-dist_sq / s2);
            sum_val += img[c * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x] * weight;
            sum_weight += weight;
        }
    }
    return sum_val / (sum_weight + 1e-6f);
}

// Forward pass
void forward(Network& net, const float* img, float* eyes_out, float* hidden, float* logits) {
    for (int i = 0; i < NUM_EYES; ++i) {
        for (int c = 0; c < CHANNELS; ++c) {
            eyes_out[i * CHANNELS + c] = sample_eye(img, c, net.eyes[i].x, net.eyes[i].y, net.eyes[i].sigma);
        }
    }
    
    // Hidden = ReLU(W1 * eyes_out + b1)
    vDSP_mmul(net.W1.data(), 1, eyes_out, 1, hidden, 1, HIDDEN_DIM, 1, INPUT_DIM);
    for(int i=0; i<HIDDEN_DIM; ++i) {
        hidden[i] += net.b1[i];
        if (hidden[i] < 0) hidden[i] = 0;
    }
    
    // Logits = W2 * hidden + b2
    vDSP_mmul(net.W2.data(), 1, hidden, 1, logits, 1, NUM_CLASSES, 1, HIDDEN_DIM);
    for(int i=0; i<NUM_CLASSES; ++i) logits[i] += net.b2[i];
}

// Load CIFAR-10 binary data
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

int main() {
    cout << "Training Gaussian Eyes on CIFAR-10 for 120s..." << endl;
    
    vector<float> train_images;
    vector<uint8_t> train_labels;
    for (int i = 1; i <= 5; ++i) {
        string path = "../amx/cifar-10-batches-bin/data_batch_" + to_string(i) + ".bin";
        if (!load_cifar(path, train_images, train_labels)) {
            cerr << "Failed to load " << path << endl;
            return 1;
        }
    }
    
    Network net;
    init_network(net);
    
    auto start_time = chrono::high_resolution_clock::now();
    int epoch = 0;
    float lr = 0.001f;
    
    // Buffers
    vector<float> eyes_out(INPUT_DIM);
    vector<float> hidden(HIDDEN_DIM);
    vector<float> logits(NUM_CLASSES);
    vector<float> d_logits(NUM_CLASSES);
    vector<float> d_hidden(HIDDEN_DIM);
    vector<float> d_eyes(INPUT_DIM);
    
    while (true) {
        auto now = chrono::high_resolution_clock::now();
        if (chrono::duration_cast<chrono::seconds>(now - start_time).count() >= TRAIN_LIMIT_S) break;
        
        float total_loss = 0;
        int correct = 0;
        
        for (int i = 0; i < 1000; ++i) { // Stochastic GD for speed
            int idx = rand() % train_labels.size();
            const float* img = &train_images[idx * 3072];
            int label = train_labels[idx];
            
            forward(net, img, eyes_out.data(), hidden.data(), logits.data());
            
            // Softmax + Cross Entropy
            float max_logit = -1e10;
            for(float l : logits) if(l > max_logit) max_logit = l;
            float sum_exp = 0;
            for(int j=0; j<NUM_CLASSES; ++j) {
                logits[j] = expf(logits[j] - max_logit);
                sum_exp += logits[j];
            }
            for(int j=0; j<NUM_CLASSES; ++j) logits[j] /= sum_exp;
            
            total_loss -= logf(logits[label] + 1e-9f);
            int pred = 0;
            for(int j=1; j<NUM_CLASSES; ++j) if(logits[j] > logits[pred]) pred = j;
            if(pred == label) correct++;
            
            // Backprop (Slightly simplified for eyes)
            for(int j=0; j<NUM_CLASSES; ++j) d_logits[j] = logits[j] - (j == label ? 1.0f : 0.0f);
            
            // Update W2, b2, and compute d_hidden
            for(int h=0; h<HIDDEN_DIM; ++h) {
                d_hidden[h] = 0;
                for(int c=0; c<NUM_CLASSES; ++c) {
                    net.W2[c * HIDDEN_DIM + h] -= lr * d_logits[c] * hidden[h];
                    d_hidden[h] += d_logits[c] * net.W2[c * HIDDEN_DIM + h];
                }
            }
            // ReLU grad
            for(int h=0; h<HIDDEN_DIM; ++h) if(hidden[h] <= 0) d_hidden[h] = 0;
            
            // Update W1, b1, and compute d_eyes
            for(int e=0; e<INPUT_DIM; ++e) {
                d_eyes[e] = 0;
                for(int h=0; h<HIDDEN_DIM; ++h) {
                    net.W1[h * INPUT_DIM + e] -= lr * d_hidden[h] * eyes_out[e];
                    d_eyes[e] += d_hidden[h] * net.W1[h * INPUT_DIM + e];
                }
            }
            
            // Eye parameter update (Numerical gradient approximation for simplicity in 120s limit)
            float eps = 0.01f;
            for (int e = 0; e < NUM_EYES; ++e) {
                // Approximate gradient wrt x
                float old_x = net.eyes[e].x;
                net.eyes[e].x += eps;
                float s_plus = 0; 
                for(int c=0; c<CHANNELS; ++c) s_plus += sample_eye(img, c, net.eyes[e].x, net.eyes[e].y, net.eyes[e].sigma);
                net.eyes[e].x = old_x;
                
                // dL/dx approx = dL/dS * dS/dx
                float dS_dx = (s_plus - (eyes_out[e*3]+eyes_out[e*3+1]+eyes_out[e*3+2])) / eps;
                net.eyes[e].x -= lr * (d_eyes[e*3]+d_eyes[e*3+1]+d_eyes[e*3+2]) * dS_dx * 0.1f;
            }
        }
        
        cout << "Epoch " << epoch++ << " | Loss: " << total_loss / 1000.0f << " | Acc: " << correct / 10.0f << "%" << endl;
    }
    
    return 0;
}
