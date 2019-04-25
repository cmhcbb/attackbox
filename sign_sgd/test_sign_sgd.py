from OPT_attack import OPT_attack
from OPT_attack_polar import OPT_attack_polar
from OPT_attack_polar_GD import OPT_attack_polar_GD
from OPT_attack_lsq import OPT_attack_lsq
from OPT_attack_polar_lsq import OPT_attack_polar_lsq
from OPT_attack_GD import OPT_attack_GD
from OPT_attack_sign_SGD import OPT_attack_sign_SGD
from models import PytorchModel
import torch
from allmodels import MNIST, load_model, load_mnist_data, load_cifar10_data, CIFAR10
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_adv(X, adv, index=0, fig=0):
    plt.subplot(1,2,1)
    plt.imshow(X[0][0], cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(adv.cpu()[0][0], cmap='gray')
    plt.savefig("results/{0}_sign_{1}.png".format(fig, index))


net = MNIST()
net.cuda()
net = torch.nn.DataParallel(net, device_ids=[0])
# print(net)
load_model(net,'../mnist_gpu.pt')
net.eval()
model = net.module if torch.cuda.is_available() else net
# model = model.cpu()
train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()

X0, Y0 = None, None
X1, Y1 = None, None
X2, Y2 = None, None
for i, (xi,yi) in enumerate(test_loader):
    if i==0:
        X0, Y0 = xi, yi
    if i==1:
        X1, Y1 = xi, yi
    if i==2:
        X2, Y2 = xi, yi
        
amodel = PytorchModel(model, bounds=[0,1], num_classes=10)

# These distortions were calculated by taking average of 10 trials for each example
# using the randomized gradient free method.
d0 = (1.5292 + 1.5095 + 1.4743 + 1.3108 + 1.2816 + 1.4416 + 1.5323 + 1.3404 + 1.3365 + 1.3819)/10
d1 = (1.2531 + 1.3640 + 1.2929 + 1.2826 + 1.2985 + 1.3726 + 1.3016 + 1.2917 + 1.2976 + 1.2911)/10
d2 = (0.7499 + 0.7520 + 0.7615 + 0.7599 + 0.7609 + 0.7635 + 0.7450 + 0.7497 + 0.8018 + 0.7475)/10
print("d0 ", d0)
print("d1 ", d1)
print("d3 ", d2)

trials = 10
print("*"*30, " Example 1 ", "*"*30)
for i in range(trials):
    attack = OPT_attack_sign_SGD(amodel)
    adv_sign, dist_sign = attack(X0.cuda(), Y0.cuda(), distortion=d0)
    print("*"*150)
    plot_adv(X0, adv_sign, index=i, fig=0)

print("*"*30, " Example 2 ", "*"*30)
for i in range(trials):
    attack = OPT_attack_sign_SGD(amodel)
    adv_sign, dist_sign = attack(X1.cuda(), Y1.cuda(), distortion=d1)
    print("*"*150)
    plot_adv(X1, adv_sign, index=i, fig=1)

print("*"*30, " Example 3 ", "*"*30)
for i in range(trials):
    attack = OPT_attack_sign_SGD(amodel)
    adv_sign, dist_sign = attack(X2.cuda(), Y2.cuda(), distortion=d2)
    print("*"*150)
    plot_adv(X2, adv_sign, index=i, fig=2)
