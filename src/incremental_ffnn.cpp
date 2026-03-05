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
const int HIDDEN_DIM = 64;
const int NUM_CLASSES = 10;
const int TRAIN_LIMIT_S = 120;

struct Network {
    vector<float> W1, b1, W2, b2;
    vector<float> mW1, vW1, mb1, vb1, mW2, vW2, mb2, vb2;
    int t = 0;
};

void init_net(Network& net) {
    mt19937 gen(42);
    float s1 = sqrtf(2.0f / INPUT_DIM);
    float s2 = sqrtf(2.0f / HIDDEN_DIM);
    normal_distribution<float> d1(0, s1), d2(0, s2);
    
    net.W1.resize(HIDDEN_DIM * INPUT_DIM);
    for(auto& w : net.W1) w = d1(gen);
    net.b1.resize(HIDDEN_DIM, 0);
    
    net.W2.resize(NUM_CLASSES * HIDDEN_DIM);
    for(auto& w : net.W2) w = d2(gen);
    net.b2.resize(NUM_CLASSES, 0);
    
    net.mW1.resize(net.W1.size(), 0); net.vW1.resize(net.W1.size(), 0);
    net.mb1.resize(net.b1.size(), 0); net.vb1.resize(net.b1.size(), 0);
    net.mW2.resize(net.W2.size(), 0); net.vW2.resize(net.W2.size(), 0);
    net.mb2.resize(net.b2.size(), 0); net.vb2.resize(net.b2.size(), 0);
}

void adam_update(vector<float>& w, vector<float>& m, vector<float>& v, const vector<float>& g, int t, float lr) {
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float lr_t = lr * sqrtf(1.0f - powf(b2, t)) / (1.0f - powf(b1, t));
    for (size_t i = 0; i < w.size(); ++i) {
        m[i] = b1 * m[i] + (1.0f - b1) * g[i];
        v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
        w[i] -= lr_t * m[i] / (sqrtf(v[i]) + eps);
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
    cout << "Incremental FFNN Training on CIFAR-10 (120s)..." << endl;
    vector<float> all_images;
    vector<uint8_t> all_labels;
    for(int i=1; i<=5; ++i) load_cifar("../amx/cifar-10-batches-bin/data_batch_" + to_string(i) + ".bin", all_images, all_labels);
    
    Network net; init_net(net);
    
    mt19937 gen(1234);
    vector<int> indices;
    for(int i=0; i<all_labels.size(); ++i) indices.push_back(i);
    shuffle(indices.begin(), indices.end(), gen);
    
    vector<int> current_subset;
    for(int i=0; i<64; ++i) current_subset.push_back(indices[i]);
    int next_idx = 64;
    
    auto start_time = chrono::high_resolution_clock::now();
    float lr = 0.001f;
    
    while (true) {
        auto now = chrono::high_resolution_clock::now();
        if (chrono::duration_cast<chrono::seconds>(now - start_time).count() >= TRAIN_LIMIT_S) break;
        
        int correct = 0;
        int subset_size = current_subset.size();
        
        // Forward + Backward on the whole subset
        vector<float> dW1(net.W1.size(), 0), db1(net.b1.size(), 0);
        vector<float> dW2(net.W2.size(), 0), db2(net.b2.size(), 0);
        
        for (int idx : current_subset) {
            const float* img = &all_images[idx * 3072];
            int label = all_labels[idx];
            
            vector<float> h(HIDDEN_DIM);
            vDSP_mmul(net.W1.data(), 1, img, 1, h.data(), 1, HIDDEN_DIM, 1, INPUT_DIM);
            for(int i=0; i<HIDDEN_DIM; ++i) { h[i] += net.b1[i]; if(h[i]<0) h[i]=0; }
            
            vector<float> logits(NUM_CLASSES);
            vDSP_mmul(net.W2.data(), 1, h.data(), 1, logits.data(), 1, NUM_CLASSES, 1, HIDDEN_DIM);
            for(int i=0; i<NUM_CLASSES; ++i) logits[i] += net.b2[i];
            
            float max_l = -1e10; for(float l : logits) if(l>max_l) max_l=l;
            float sum_e = 0; for(float& l : logits) { l=expf(l-max_l); sum_e+=l; }
            int pred = 0;
            for(int i=0; i<NUM_CLASSES; ++i) {
                logits[i]/=sum_e;
                if(logits[i] > logits[pred]) pred = i;
            }
            if(pred == label) correct++;
            
            // Grads
            vector<float> dLogits(NUM_CLASSES);
            for(int i=0; i<NUM_CLASSES; ++i) dLogits[i] = logits[i] - (i == label ? 1.0f : 0.0f);
            
            for(int i=0; i<NUM_CLASSES; ++i) {
                db2[i] += dLogits[i];
                for(int j=0; j<HIDDEN_DIM; ++j) dW2[i*HIDDEN_DIM+j] += dLogits[i] * h[j];
            }
            
            vector<float> dh(HIDDEN_DIM, 0);
            for(int i=0; i<HIDDEN_DIM; ++i) {
                if(h[i] > 0) {
                    for(int j=0; j<NUM_CLASSES; ++j) dh[i] += dLogits[j] * net.W2[j*HIDDEN_DIM+i];
                }
            }
            
            for(int i=0; i<HIDDEN_DIM; ++i) {
                db1[i] += dh[i];
                for(int j=0; j<INPUT_DIM; ++j) dW1[i*INPUT_DIM+j] += dh[i] * img[j];
            }
        }
        
        if (correct == subset_size) {
            cout << "Subset size " << subset_size << " reached 100% accuracy. Adding 4 more images." << endl;
            for(int i=0; i<4 && next_idx < indices.size(); ++i) current_subset.push_back(indices[next_idx++]);
            continue; 
        }
        
        // Update
        net.t++;
        float inv_N = 1.0f / subset_size;
        for(auto& g : dW1) g *= inv_N; for(auto& g : db1) g *= inv_N;
        for(auto& g : dW2) g *= inv_N; for(auto& g : db2) g *= inv_N;
        
        adam_update(net.W1, net.mW1, net.vW1, dW1, net.t, lr);
        adam_update(net.b1, net.mb1, net.vb1, db1, net.t, lr);
        adam_update(net.W2, net.mW2, net.vW2, dW2, net.t, lr);
        adam_update(net.b2, net.mb2, net.vb2, db2, net.t, lr);
        
        if (net.t % 100 == 0) {
            cout << "Iter " << net.t << " | Subset: " << subset_size << " | Acc: " << (float)correct/subset_size*100 << "%" << endl;
        }
    }
    
    return 0;
}
