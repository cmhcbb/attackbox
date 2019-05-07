from CW import CW
from FGSM import FGSM
from OPT_attack import OPT_attack
from OPT_attack_polar import OPT_attack_polar
from OPT_attack_polar_GD import OPT_attack_polar_GD
from OPT_attack_lsq import OPT_attack_lsq
from OPT_attack_polar_lsq import OPT_attack_polar_lsq
from OPT_attack_GD import OPT_attack_GD
from OPT_genattack import OPT_genattack
from OPT_attack_sign_SGD import OPT_attack_sign_SGD
from ZOO import ZOO
from OPT_attack_lf import OPT_attack_lf
from nes_attack import NES
from models import PytorchModel
import torch
from allmodels import MNIST, load_model, load_mnist_data, load_cifar10_data, CIFAR10
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1";

def plot_adv(X, adv, index=0, fig=0, filename="randomized"):
    plt.subplot(1,2,1)
    plt.imshow(np.transpose(X[0].numpy(), (1, 2, 0)))
    plt.subplot(1,2,2)
    plt.imshow(np.transpose(adv.cpu()[0].numpy(), (1, 2, 0)))
    plt.savefig("results/cifar10/{0}_{1}_{2}.png".format(fig, filename, index))


net = CIFAR10()
net.cuda()
net = torch.nn.DataParallel(net, device_ids=[0])
# print(net)
load_model(net,'cifar10_gpu.pt')
net.eval()
model = net.module if torch.cuda.is_available() else net
# model = model.cpu()
train_loader, test_loader, train_dataset, test_dataset = load_cifar10_data()

X0, Y0 = None, None
X1, Y1 = None, None
X2, Y2 = None, None
for i, (xi,yi) in enumerate(test_loader):
    if i==2:
        X0, Y0 = xi, yi
    if i==4:
        X1, Y1 = xi, yi
    if i==7:
        X2, Y2 = xi, yi
    if i==10:
        break
        
amodel = PytorchModel(model, bounds=[0,1], num_classes=10)

trials = 3

example = 0
for x,y in [(X0, Y0), (X1,Y1), (X2, Y2)]:
    print("*"*30, " Example {0} ".format(example), "*"*30)
    for i in range(trials):
        print("Trial ", i)
        seed = np.random.randint(1000)
        
        # Randomized
        attack = OPT_attack(amodel)
        adv_rand, dist_rand = attack(x.cuda(), y.cuda(), seed=seed)
        plot_adv(x, adv_rand, index=i, fig=example, filename="randomized")
        
        print("*"*80)

        # Sign SGD
        attack_sign = OPT_attack_sign_SGD(amodel)
        adv_sign, dist_sign = attack_sign(x.cuda(), y.cuda(), seed=seed)
        plot_adv(x, adv_sign, index=i, fig=example, filename="sign")
    
    example += 1