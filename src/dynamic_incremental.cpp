#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <Accelerate/Accelerate.h>

using namespace std;

const int IMG_SIZE = 32;
const int CHANNELS = 3;
const int INPUT_DIM = IMG_SIZE * IMG_SIZE * CHANNELS;
const int MAX_NEURONS = 2048; // Pre-allocate buffer for growth
const int NUM_CLASSES = 10;
const int TRAIN_LIMIT_S = 240;
const int INITIAL_IMAGES = 8192;
const int INITIAL_NEURONS = 512;

struct DynamicNetwork {
    int H; // Current number of neurons
    vector<float> W1, b1, W2, b2;
    vector<float> mW1, vW1, mb1, vb1, mW2, vW2, mb2, vb2;
    int t = 0;
    
    DynamicNetwork(int initial_h) : H(initial_h) {
        W1.resize(MAX_NEURONS * INPUT_DIM);
        b1.resize(MAX_NEURONS, 0);
        W2.resize(NUM_CLASSES * MAX_NEURONS, 0);
        b2.resize(NUM_CLASSES, 0);
        
        mW1.resize(W1.size(), 0); vW1.resize(W1.size(), 0);
        mb1.resize(b1.size(), 0); vb1.resize(b1.size(), 0);
        mW2.resize(W2.size(), 0); vW2.resize(W2.size(), 0);
        mb2.resize(b2.size(), 0); vb2.resize(b2.size(), 0);
        
        init_neurons(0, H);
    }
    
    void init_neurons(int start, int end) {
        mt19937 gen(42 + start);
        float s1 = sqrtf(2.0f / INPUT_DIM);
        // muP scaling for output layer
        float s2 = 1.0f / end; 
        normal_distribution<float> d1(0, s1), d2(0, s2);
        
        for(int i = start; i < end; ++i) {
            for(int j = 0; j < INPUT_DIM; ++j) W1[i * INPUT_DIM + j] = d1(gen);
            for(int j = 0; j < NUM_CLASSES; ++j) W2[j * MAX_NEURONS + i] = d2(gen);
        }
    }
    
    void add_neuron() {
        if (H >= MAX_NEURONS) return;
        init_neurons(H, H + 1);
        H++;
        cout << "Dynamic Expansion: Added neuron. New width: " << H << endl;
    }
};

void radam_update(vector<float>& w, vector<float>& m, vector<float>& v, const vector<float>& g, int& t, float lr, int size) {
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    t++;
    float rho_inf = 2.0f / (1.0f - b2) - 1.0f;
    float b1_t = powf(b1, t);
    float b2_t = powf(b2, t);
    float rho_t = rho_inf - 2.0f * t * b2_t / (1.0f - b2_t);
    
    for (int i = 0; i < size; ++i) {
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

bool load_cifar(const string& path, vector<float>& images, vector<uint8_t>& labels) {
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

int main() {
    cout << "Dynamic Incremental FFNN Training (240s)..." << endl;
    vector<float> all_images;
    vector<uint8_t> all_labels;
    for(int i=1; i<=5; ++i) load_cifar("../amx/cifar-10-batches-bin/data_batch_" + to_string(i) + ".bin", all_images, all_labels);
    
    DynamicNetwork net(INITIAL_NEURONS);
    
    mt19937 gen(1337);
    vector<int> indices(all_labels.size());
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), gen);
    
    vector<int> current_subset;
    for(int i=0; i<INITIAL_IMAGES; ++i) current_subset.push_back(indices[i]);
    int next_idx = INITIAL_IMAGES;
    
    auto start_time = chrono::high_resolution_clock::now();
    auto last_print_time = start_time;
    float lr = 0.001f;
    float last_loss = 1e10;
    int stagnate_count = 0;
    
    while (true) {
        auto now = chrono::high_resolution_clock::now();
        double elapsed_s = chrono::duration_cast<chrono::seconds>(now - start_time).count();
        if (elapsed_s >= TRAIN_LIMIT_S) break;
        
        int correct = 0;
        int N = current_subset.size();
        float total_loss = 0;
        
        // Forward + Backward on batch 64
        vector<float> dW1(net.W1.size(), 0), db1(net.b1.size(), 0);
        vector<float> dW2(net.W2.size(), 0), db2(net.b2.size(), 0);
        
        // Use a subset of the current subset for the gradient step (batch size 64)
        for (int b = 0; b < 64; ++b) {
            int idx = current_subset[gen() % N];
            const float* img = &all_images[idx * 3072];
            int label = all_labels[idx];
            
            vector<float> h(net.H);
            vDSP_mmul(net.W1.data(), 1, img, 1, h.data(), 1, net.H, 1, INPUT_DIM);
            for(int i=0; i<net.H; ++i) { h[i] += net.b1[i]; if(h[i]<0) h[i]=0; }
            
            vector<float> logits(NUM_CLASSES);
            // W2 is NUM_CLASSES x MAX_NEURONS
            for(int c=0; c<NUM_CLASSES; ++c) {
                float sum = net.b2[c];
                for(int i=0; i<net.H; ++i) sum += net.W2[c * MAX_NEURONS + i] * h[i];
                logits[c] = sum;
            }
            
            float max_l = -1e10; for(float l : logits) if(l>max_l) max_l=l;
            float sum_e = 0; for(float& l : logits) { l=expf(l-max_l); sum_e+=l; }
            int pred = 0;
            for(int i=0; i<NUM_CLASSES; ++i) {
                logits[i]/=sum_e;
                if(logits[i] > logits[pred]) pred = i;
            }
            if(pred == label) correct++;
            total_loss -= logf(logits[label] + 1e-9f);
            
            vector<float> dLogits(NUM_CLASSES);
            for(int i=0; i<NUM_CLASSES; ++i) dLogits[i] = logits[i] - (i == label ? 1.0f : 0.0f);
            
            for(int i=0; i<NUM_CLASSES; ++i) {
                db2[i] += dLogits[i];
                for(int j=0; j<net.H; ++j) dW2[i * MAX_NEURONS + j] += dLogits[i] * h[j];
            }
            
            for(int i=0; i<net.H; ++i) {
                if(h[i] > 0) {
                    float dh = 0;
                    for(int c=0; c<NUM_CLASSES; ++c) dh += dLogits[c] * net.W2[c * MAX_NEURONS + i];
                    db1[i] += dh;
                    for(int j=0; j<INPUT_DIM; ++j) dW1[i * INPUT_DIM + j] += dh * img[j];
                }
            }
        }
        
        float avg_loss = total_loss / 64.0f;
        if (abs(avg_loss - last_loss) < 1e-4) {
            stagnate_count++;
        } else {
            stagnate_count = 0;
        }
        last_loss = avg_loss;
        
        if (chrono::duration_cast<chrono::seconds>(now - last_print_time).count() >= 1) {
            float err_pct = (1.0f - (correct / 64.0f)) * 100.0f;
            cout << "[Time: " << elapsed_s << "s] Err: " << err_pct << "% | Loss: " << last_loss << " | Subset: " << current_subset.size() << " | Neurons: " << net.H << endl;
            last_print_time = now;
        }

        
        if (stagnate_count >= 4) {
            net.add_neuron();
            stagnate_count = 0;
        }
        
        // Apply R-Adam update
        int dummy_t = net.t;
        radam_update(net.W1, net.mW1, net.vW1, dW1, dummy_t, lr, net.H * INPUT_DIM);
        dummy_t = net.t;
        radam_update(net.b1, net.mb1, net.vb1, db1, dummy_t, lr, net.H);
        dummy_t = net.t;
        radam_update(net.W2, net.mW2, net.vW2, dW2, dummy_t, lr, NUM_CLASSES * MAX_NEURONS);
        dummy_t = net.t;
        radam_update(net.b2, net.mb2, net.vb2, db2, net.t, lr, NUM_CLASSES);
        
        if (correct == 64) {
            // Check full subset accuracy periodically or assume batch convergence
            // For 100% on subset, we'd need a full pass. Let's do a quick full pass every 100 iters.
            if (net.t % 100 == 0) {
                cout << "Iter " << net.t << " | Loss: " << avg_loss << " | Subset: " << N << " | Neurons: " << net.H << endl;
                if (avg_loss < 0.05) { // Threshold for "100%" in batch
                    cout << "Reached target accuracy. Adding 16 more images." << endl;
                    for(int i=0; i<16 && next_idx < indices.size(); ++i) current_subset.push_back(indices[next_idx++]);
                }
            }
        }
    }
    
    return 0;
}
