from CW import CW
from FGSM import FGSM
from OPT_attack import OPT_attack
from OPT_genattack import OPT_genattack
from ZOO import ZOO
from OPT_attack_lf import OPT_attack_lf
from Sign_OPT import OPT_attack_sign_SGD
from Sign_OPT_lf import OPT_attack_sign_SGD_lf
from NES import NES
from PGD import PGD
from models import PytorchModel
import torch
from allmodels import MNIST, load_model, load_mnist_data, load_cifar10_data, CIFAR10, VGG_plain, VGG_rse, VGG_vi
import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="MNIST",
                    help='Dataset to be used, [MNIST, CIFAR10, Imagenet]')
parser.add_argument('--attack', type=str, default=None,
                    help='Attack to be used')

parser.add_argument('--n_neigh', type=int, default=0,
                    help='number of neighbors of target node')
parser.add_argument('--start', type=int, default=0,
                    help='starting node')
parser.add_argument('--npoints', type=int, default=10,
                    help='points to be added')
parser.add_argument('--hops', type=int, default=1,
                    help='hops of neighbors of target node')
parser.add_argument('--epsilon', type=float, default=0.01,
                    help='epsilon in the PGD attack')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='verbose.')
parser.add_argument('--test_batch_size', type=int, default=1,
                    help='test batch_size')
args = parser.parse_args()
#np.random.seed(args.seed)
#torch.manual_seed(args.seed)



if args.dataset == "MNIST":
    net = MNIST()
    net = torch.nn.DataParallel(net, device_ids=[0])
    load_model(net,'mnist_gpu.pt')
    train_loader, test_loader, train_dataset, test_dataset = load_mnist_data(args.test_batch_size)
elif args.dataset == 'CIFAR':
    net = CIFAR10() 
    net = torch.nn.DataParallel(net, device_ids=[0])
    load_model(net, 'cifar10_gpu.pt')
    train_loader, test_loader, train_dataset, test_dataset = load_cifar10_data(args.test_batch_size)
elif args.dataset == 'Imagenet':
    net = CIFAR10() 
    net = torch.nn.DataParallel(net, device_ids=[0])
    load_model(net, 'cifar10_gpu.pt')
else:
    print("Unsupport dataset")
    os.exit(0)

attack_list = {
    "PGD":PGD,
    "Sign_OPT": OPT_attack_sign_SGD,
    "Sign_OPT_lf": OPT_attack_sign_SGD_lf,
    "CW": CW,
    "OPT_attack": OPT_attack,
    "OPT_attack_lf": OPT_attack_lf,
    "FGSM": FGSM,
    "NES": NES,
    "ZOO": ZOO
}


net.cuda()
net.eval()
#net = VGG_rse('VGG16', 10, 0.2,0.1, img_width=32)
#net = VGG_plain('VGG16', 10, img_width=32)
#net.cuda()
#net = torch.nn.DataParallel(net, device_ids=[0])
#load_model(net,'./defense_model/cifar10_vgg_plain.pth')
#net.eval()
#model = net.module if torch.cuda.is_available() else net
#net = CIFAR10() 
#net = torch.nn.DataParallel(net, device_ids=[0])
#load_model(net, 'cifar10_gpu.pt')
#net.eval()
#net.cuda()
model = net.module if torch.cuda.is_available() else net



amodel = PytorchModel(model, bounds=[0,1], num_classes=10)
attack = attack_list[args.attack](amodel)
#attack = CW(amodel)
#attack = FGSM(amodel)
#attack = OPT_attack(amodel)
#attack = OPT_attack_sign_SGD_lf(amodel)
#attack = OPT_genattack(amodel) 
#attack = OPT_attack(amodel)  
#attack = NES(amodel)
#attack = ZOO(amodel)
#attack = PGD(amodel)
#attack = OPT_attack_sign_SGD(amodel)

#train_loader, test_loader, train_dataset, test_dataset = load_mnist_data(args.test_batch_size)
#train_loader, test_loader, train_dataset, test_dataset = load_cifar10_data()
for i, (xi,yi) in enumerate(test_loader):
    print("image "+str(i))
    if i==1:
        #continue
        break
    xi,yi = xi.cuda(), yi.cuda()
    #if i==3:
    #amodel.predict_ensemble(xi)
    #adv=attack(xi,yi, 0.2)
    adv=attack(xi,yi)
    #r_count= (torch.max(amodel.predict(adv),1)[1]==yi).nonzero().shape[0]
    #clean_count= (torch.max(amodel.predict(xi),1)[1]==yi).nonzero().shape[0]
    #total_r_count += r_count
    #print(clean_count - r_count, (clean_count - r_count)/clean_count)
