import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4";

from OPT_attack import OPT_attack
from OPT_attack_lf import OPT_attack_lf
from OPT_attack_sign_SGD import OPT_attack_sign_SGD
from models import PytorchModel
import torch
from allmodels import MNIST, load_model, load_mnist_data, load_cifar10_data, CIFAR10
import numpy as np
import matplotlib.pyplot as plt



net = MNIST()
net.cuda()
net = torch.nn.DataParallel(net, device_ids=[0])
load_model(net,'mnist_gpu.pt')
net.eval()
model = net.module if torch.cuda.is_available() else net
train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()
amodel = PytorchModel(model, bounds=[0,1], num_classes=10)

train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()

attack = OPT_attack_lf(amodel, train_dataset=train_dataset)

np.random.seed(0)
seeds = np.random.randint(10000, size=[50])
print(seeds)

for i, (xi,yi) in enumerate(test_loader):
    if i==50:
        break
    np.random.seed(seeds[i])
    target = np.random.randint(10)
    print("Attacking Source: {0} Target: {1} Seed: {2} Number {3}".format(yi.item(), target, seeds[i], i))
    adv, dist = attack(xi.cuda(), yi.cuda(), target=target*torch.ones(1, dtype=torch.long).cuda(), seed=seeds[i])
    print()
