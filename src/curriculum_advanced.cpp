#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <Accelerate/Accelerate.h>

using namespace std;

const int IMG_SIZE = 32;
const int CHANNELS = 3;
const int INPUT_DIM = IMG_SIZE * IMG_SIZE * CHANNELS;
const int MAX_NEURONS = 4096;
const int NUM_CLASSES = 10;
const int TRAIN_LIMIT_S = 600; 
const int INITIAL_IMAGES = 8192;
const int INITIAL_NEURONS = 64;
const int BATCH_SIZE = 64;

struct DynamicNetwork {
    int H; 
    vector<float> W1, b1, W2, b2;
    vector<float> bn_gamma, bn_beta;
    vector<float> mW1, vW1, mb1, vb1, mW2, vW2, mb2, vb2;
    vector<float> mG, vG, mB, vB;
    int t = 0;
    
    DynamicNetwork(int initial_h) : H(initial_h) {
        W1.resize(MAX_NEURONS * INPUT_DIM);
        b1.resize(MAX_NEURONS, 0);
        W2.resize(NUM_CLASSES * MAX_NEURONS, 0);
        b2.resize(NUM_CLASSES, 0);
        bn_gamma.resize(MAX_NEURONS, 1.0f);
        bn_beta.resize(MAX_NEURONS, 0.0f);
        
        mW1.resize(W1.size(), 0); vW1.resize(W1.size(), 0);
        mb1.resize(b1.size(), 0); vb1.resize(b1.size(), 0);
        mW2.resize(W2.size(), 0); vW2.resize(W2.size(), 0);
        mb2.resize(b2.size(), 0); vb2.resize(b2.size(), 0);
        mG.resize(MAX_NEURONS, 0); vG.resize(MAX_NEURONS, 0);
        mB.resize(MAX_NEURONS, 0); vB.resize(MAX_NEURONS, 0);
        
        init_neurons(0, H);
    }
    
    void init_neurons(int start, int end) {
        mt19937 gen(42 + start);
        float s1 = sqrtf(2.0f / INPUT_DIM);
        float s2 = 1.0f / end; 
        normal_distribution<float> d1(0, s1), d2(0, s2);
        for(int i = start; i < end; ++i) {
            for(int j = 0; j < INPUT_DIM; ++j) W1[i * INPUT_DIM + j] = d1(gen);
            for(int j = 0; j < NUM_CLASSES; ++j) W2[j * MAX_NEURONS + i] = d2(gen);
        }
    }
    
    void add_neurons(int count) {
        if (H + count > MAX_NEURONS) return;
        init_neurons(H, H + count);
        H += count;
        cout << "[Expansion] Added " << count << " neurons. Width: " << H << endl;
    }
};

void radam_update(vector<float>& w, vector<float>& m, vector<float>& v, const vector<float>& g, int& t, float lr, int size) {
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
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

float l2_dist_sq(const float* a, const float* b) {
    float sum = 0;
    for(int i=0; i<3072; ++i) { float d = a[i] - b[i]; sum += d*d; }
    return sum;
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
    cout << "Advanced BN + Diversity Curriculum Training..." << endl;
    vector<float> all_images;
    vector<uint8_t> all_labels;
    for(int i=1; i<=5; ++i) load_cifar("../amx/cifar-10-batches-bin/data_batch_" + to_string(i) + ".bin", all_images, all_labels);
    
    DynamicNetwork net(INITIAL_NEURONS);
    mt19937 gen(1337);
    vector<int> indices(all_labels.size());
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), gen);
    
    vector<int> current_subset;
    current_subset.push_back(indices[0]);
    for(int i=1; i<INITIAL_IMAGES; ++i) {
        int best_cand = -1; float max_dist = -1;
        for(int j=0; j<10; ++j) {
            int cand = indices[INITIAL_IMAGES + (gen() % (indices.size() - INITIAL_IMAGES))];
            float d = l2_dist_sq(&all_images[current_subset.back() * 3072], &all_images[cand * 3072]);
            if(d > max_dist) { max_dist = d; best_cand = cand; }
        }
        current_subset.push_back(best_cand);
    }
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
        int N_sub = current_subset.size();
        float total_loss = 0;
        
        vector<float> dW1(net.W1.size(), 0), db1(net.b1.size(), 0);
        vector<float> dW2(net.W2.size(), 0), db2(net.b2.size(), 0);
        vector<float> dG(MAX_NEURONS, 0), dB(MAX_NEURONS, 0);
        
        vector<int> batch_indices;
        for(int i=0; i<BATCH_SIZE; ++i) batch_indices.push_back(current_subset[gen() % N_sub]);
        
        // Forward
        vector<vector<float>> hs(BATCH_SIZE, vector<float>(net.H));
        vector<vector<float>> hs_norm(BATCH_SIZE, vector<float>(net.H));
        vector<float> mu(net.H, 0), var(net.H, 0);
        
        for (int b = 0; b < BATCH_SIZE; ++b) {
            vDSP_mmul(net.W1.data(), 1, &all_images[batch_indices[b] * 3072], 1, hs[b].data(), 1, net.H, 1, INPUT_DIM);
            for(int i=0; i<net.H; ++i) { hs[b][i] += net.b1[i]; mu[i] += hs[b][i]; }
        }
        for(int i=0; i<net.H; ++i) mu[i] /= BATCH_SIZE;
        for (int b = 0; b < BATCH_SIZE; ++b) {
            for(int i=0; i<net.H; ++i) var[i] += powf(hs[b][i] - mu[i], 2);
        }
        for(int i=0; i<net.H; ++i) var[i] = sqrtf(var[i] / BATCH_SIZE + 1e-5f);
        
        vector<vector<float>> dL_dhs(BATCH_SIZE, vector<float>(net.H, 0));
        
        for (int b = 0; b < BATCH_SIZE; ++b) {
            int label = all_labels[batch_indices[b]];
            vector<float> logits(NUM_CLASSES);
            for(int i=0; i<net.H; ++i) {
                hs_norm[b][i] = (hs[b][i] - mu[i]) / var[i];
                float act = hs_norm[b][i] * net.bn_gamma[i] + net.bn_beta[i];
                hs[b][i] = act > 0 ? act : 0;
            }
            for(int c=0; c<NUM_CLASSES; ++c) {
                float sum = net.b2[c];
                for(int i=0; i<net.H; ++i) sum += net.W2[c * MAX_NEURONS + i] * hs[b][i];
                logits[c] = sum;
            }
            float max_l = -1e10; for(float l : logits) if(l>max_l) max_l=l;
            float sum_e = 0; for(float& l : logits) { l=expf(l-max_l); sum_e+=l; }
            int pred = 0;
            for(int i=0; i<NUM_CLASSES; ++i) {
                logits[i]/=sum_e; if(logits[i] > logits[pred]) pred = i;
            }
            if(pred == label) correct++;
            total_loss -= logf(logits[label] + 1e-9f);
            
            vector<float> dLogits(NUM_CLASSES);
            for(int i=0; i<NUM_CLASSES; ++i) {
                dLogits[i] = logits[i] - (i == label ? 1.0f : 0.0f);
                db2[i] += dLogits[i];
                for(int j=0; j<net.H; ++j) {
                    float dW = dLogits[i] * hs[b][j];
                    dW2[i * MAX_NEURONS + j] += dW;
                    if(hs[b][j] > 0) dL_dhs[b][j] += dLogits[i] * net.W2[i * MAX_NEURONS + j];
                }
            }
        }
        
        // Full BN Backprop + W1/b1 update
        for(int i=0; i<net.H; ++i) {
            float sum_dl_dy = 0, sum_dl_dy_xhat = 0;
            for(int b=0; b<BATCH_SIZE; ++b) {
                float dl_dy = dL_dhs[b][i] * net.bn_gamma[i];
                sum_dl_dy += dl_dy;
                sum_dl_dy_xhat += dl_dy * hs_norm[b][i];
                dG[i] += dL_dhs[b][i] * hs_norm[b][i];
                dB[i] += dL_dhs[b][i];
            }
            for(int b=0; b<BATCH_SIZE; ++b) {
                float dl_dy = dL_dhs[b][i] * net.bn_gamma[i];
                float dl_dx = (1.0f / (BATCH_SIZE * var[i])) * (BATCH_SIZE * dl_dy - sum_dl_dy - hs_norm[b][i] * sum_dl_dy_xhat);
                db1[i] += dl_dx;
                for(int j=0; j<INPUT_DIM; ++j) dW1[i * INPUT_DIM + j] += dl_dx * all_images[batch_indices[b] * 3072 + j];
            }
        }
        
        float avg_loss = total_loss / BATCH_SIZE;
        if (abs(avg_loss - last_loss) < 1e-5) stagnate_count++; else stagnate_count = 0;
        last_loss = avg_loss;
        
        if (chrono::duration_cast<chrono::seconds>(now - last_print_time).count() >= 1) {
            float err_pct = (1.0f - (correct / (float)BATCH_SIZE)) * 100.0f;
            cout << "[Time: " << (int)elapsed_s << "s] Err: " << err_pct << "% | Loss: " << avg_loss << " | Subset: " << N_sub << " | Neurons: " << net.H << endl;
            last_print_time = now;
        }

        if (stagnate_count >= 4) { net.add_neurons(2); stagnate_count = 0; }
        
        net.t++; float inv_B = 1.0f / BATCH_SIZE;
        for(auto& g : dW1) g *= inv_B; for(auto& g : db1) g *= inv_B;
        for(auto& g : dW2) g *= inv_B; for(auto& g : db2) g *= inv_B;
        for(auto& g : dG) g *= inv_B; for(auto& g : dB) g *= inv_B;

        radam_update(net.W1, net.mW1, net.vW1, dW1, net.t, lr, net.H * INPUT_DIM);
        int dt = net.t;
        radam_update(net.b1, net.mb1, net.vb1, db1, dt, lr, net.H); dt = net.t;
        radam_update(net.W2, net.mW2, net.vW2, dW2, dt, lr, NUM_CLASSES * MAX_NEURONS); dt = net.t;
        radam_update(net.b2, net.mb2, net.vb2, db2, dt, lr, NUM_CLASSES); dt = net.t;
        radam_update(net.bn_gamma, net.mG, net.vG, dG, dt, lr, net.H); dt = net.t;
        radam_update(net.bn_beta, net.mB, net.vB, dB, dt, lr, net.H);
        
        if (avg_loss < 0.05) {
            if (net.t % 20 == 0) {
                cout << "[Curriculum] Adding 128 images." << endl;
                for(int i=0; i<128 && next_idx < indices.size(); ++i) current_subset.push_back(indices[next_idx++]);
            }
        }
    }
    return 0;
}
