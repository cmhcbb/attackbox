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


net = CIFAR10()
net.cuda()
net = torch.nn.DataParallel(net, device_ids=[0])
load_model(net,'cifar10_gpu.pt')
net.eval()
model = net.module if torch.cuda.is_available() else net
amodel = PytorchModel(model, bounds=[0,1], num_classes=10)

train_loader, test_loader, train_dataset, test_dataset = load_cifar10_data()

attack = OPT_attack_lf(amodel)

np.random.seed(0)
seeds = np.random.randint(10000, size=[50])
print(seeds)

for i, (xi,yi) in enumerate(test_loader):
    if i==50:
        break
    print("Attacking Source: {0} Seed: {1} Number {2}".format(yi.item(), seeds[i], i))
    adv, dist = attack(xi.cuda(), yi.cuda(), seed=seeds[i])
    print()