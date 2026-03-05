import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import argparse
import numpy as np

# Constraints: 1024 width, CIFAR-10, H200 optimized
class FFNN(nn.Module):
    def __init__(self, width=1024):
        super(FFNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3072, width),
            nn.BatchNorm1d(width),
            nn.LeakyReLU(0.1),
            nn.Linear(width, 10)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)

def get_optimizer(name, model, lr=0.001):
    name = name.lower()
    if name == 'sgd': return optim.SGD(model.parameters(), lr=lr)
    elif name == 'adam': return optim.Adam(model.parameters(), lr=lr)
    elif name == 'radam': return optim.RAdam(model.parameters(), lr=lr)
    elif name == 'nesterov': return optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    elif name == 'rmsprop': return optim.RMSprop(model.parameters(), lr=lr)
    elif name == 'adagrad': return optim.Adagrad(model.parameters(), lr=lr)
    elif name == 'lbfgs': return optim.LBFGS(model.parameters(), lr=lr, history_size=10)
    elif name == 'lamb':
        # Simple LAMB implementation or fallback to AdamW if not available in standard torch
        try:
            from torch.optim import AdamW
            return AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        except:
            return optim.Adam(model.parameters(), lr=lr)
    else: raise ValueError(f"Unknown solver: {name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', type=str, default='adam', help='Solver name (sgd, adam, radam, nesterov, rmsprop, adagrad, lamb, lbfgs)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Targeting Device: {device} | Solver: {args.solver.upper()}")

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Pre-load to GPU for H200 throughput
    all_data = torch.stack([trainset[i][0] for i in range(len(trainset))]).to(device)
    all_labels = torch.tensor([trainset[i][1] for i in range(len(trainset))]).to(device)
    
    model = FFNN(1024).to(device)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: Benchmark (10s) - (Err Reduction / Time)
    print("Benchmarking Batch/Subset configurations...")
    best_config = (4096, 8192)
    best_efficiency = -1.0
    
    bench_start = time.time()
    while time.time() - bench_start < 10:
        b_size = int(np.random.normal(4096, 2048))
        s_size = int(np.random.normal(8192, 4096))
        b_size = max(128, min(s_size, b_size))
        s_size = max(b_size, min(50000, s_size))
        
        # Reset model for fair comparison
        model.layers[0].reset_parameters()
        optimizer = get_optimizer(args.solver, model)
        
        iter_start = time.time()
        idx = torch.randperm(50000)[:s_size]
        sub_data, sub_labels = all_data[idx], all_labels[idx]
        
        # Measure initial loss
        with torch.no_grad():
            logits = model(sub_data[:b_size])
            loss_init = criterion(logits, sub_labels[:b_size]).item()
        
        # One step
        def closure():
            optimizer.zero_grad()
            out = model(sub_data[:b_size])
            loss = criterion(out, sub_labels[:b_size])
            loss.backward()
            return loss
        
        if args.solver == 'lbfgs': optimizer.step(closure)
        else: closure(); optimizer.step()
        
        # Measure final loss
        with torch.no_grad():
            logits = model(sub_data[:b_size])
            loss_final = criterion(logits, sub_labels[:b_size]).item()
            correct = (logits.argmax(1) == sub_labels[:b_size]).float().mean().item()
        
        elapsed = time.time() - iter_start
        efficiency = (loss_init - loss_final) / elapsed
        
        if efficiency > best_efficiency:
            best_efficiency = efficiency
            best_config = (b_size, s_size)
            
    print(f"Winner: Batch={best_config[0]}, Subset={best_config[1]} | Efficiency: {best_efficiency:.4f} ΔLoss/s")

    # Phase 2: Train (120s)
    optimizer = get_optimizer(args.solver, model)
    train_start = time.time()
    last_print = -1
    t = 0
    
    while time.time() - train_start < 120:
        b_size, s_size = best_config
        idx = torch.randperm(50000)[:b_size]
        data, labels = all_data[idx], all_labels[idx]
        
        def closure():
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, labels)
            loss.backward()
            return loss, out

        if args.solver == 'lbfgs':
            loss_val, out = optimizer.step(lambda: closure()[0]), None # out is complex for LBFGS loop
            # Simple re-run for stats
            with torch.no_grad(): out = model(data)
        else:
            loss_val, out = closure()
            optimizer.step()
            
        t += 1
        elapsed = time.time() - train_start
        if int(elapsed) > last_print:
            correct = (out.argmax(1) == labels).float().mean().item()
            print(f"[Time: {int(elapsed)}s] Err: {(1.0-correct)*100:.2f}% | Loss: {loss_val.item():.4f} | B: {b_size}")
            last_print = int(elapsed)

if __name__ == "__main__":
    main()
