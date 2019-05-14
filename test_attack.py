from CW import CW
from FGSM import FGSM
from OPT_attack import OPT_attack
from OPT_genattack import OPT_genattack
from ZOO import ZOO
from OPT_attack_lf import OPT_attack_lf
from NES import NES
from models import PytorchModel
import torch
from allmodels import MNIST, load_model, load_mnist_data, load_cifar10_data, CIFAR10, VGG_plain, VGG_rse, VGG_vi
import os




#net = MNIST()
#net.cuda()
#net = torch.nn.DataParallel(net, device_ids=[0])
#load_model(net,'mnist_gpu.pt')
#net.eval()
net = VGG_plain('VGG16', 10, img_width=32)
net.cuda()
net = torch.nn.DataParallel(net, device_ids=[0])
load_model(net,'./defense_model/cifar10_vgg_plain.pth')
net.eval()
model = net.module if torch.cuda.is_available() else net


amodel = PytorchModel(model, bounds=[0,1], num_classes=10)
#attack = CW(amodel)
#attack = FGSM(amodel)
attack = OPT_attack(amodel)
#attack = OPT_genattack(amodel) 
#attack = OPT_attack(amodel)  
#attack = NES(amodel)
#attack = ZOO(amodel)


#train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()
train_loader, test_loader, train_dataset, test_dataset = load_cifar10_data()
for i, (xi,yi) in enumerate(test_loader):
    if i==50:
        break
    #if i==3:
    adv=attack(xi,yi)
