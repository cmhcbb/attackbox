from attack import *
from models import PytorchModel
import torch
from allmodels import MNIST, load_model, load_mnist_data, load_cifar10_data, CIFAR10, VGG_plain, VGG_rse, VGG_vi
#from wideresnet import *
import os, argparse
import numpy as np
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
parser.add_argument('--targeted', action='store_true', default=False,
                    help='Targeted attack.')
parser.add_argument('--random_start', action='store_true', default=False,
                    help='PGD attack with random start.')
parser.add_argument('--fd_eta', type=float, help='\eta, used to estimate the derivative via finite differences')
parser.add_argument('--image_lr', type=float, help='Learning rate for the image (iterative attack)')
parser.add_argument('--online_lr', type=float, help='Learning rate for the prior')
parser.add_argument('--mode', type=str, help='Which lp constraint to run bandits [linf|l2]')
parser.add_argument('--exploration', type=float, help='\delta, parameterizes the exploration to be done around the prior')
 

#parser.add_argument('--n_neigh', type=int, default=0,
#                    help='number of neighbors of target node')
#parser.add_argument('--start', type=int, default=0,
#                    help='starting node')
#parser.add_argument('--npoints', type=int, default=10,
#                    help='points to be added')
#parser.add_argument('--hops', type=int, default=1,
#                    help='hops of neighbors of target node')
parser.add_argument('--epsilon', type=float, default=0.01,
                        help='epsilon in the PGD attack')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='verbose.')
parser.add_argument('--test_batch_size', type=int, default=1,
                    help='test batch_size')
parser.add_argument('--test_batch', type=int, default=10,
                    help='test batch number')
parser.add_argument('--model_dir', type=str,  help='model loading directory')


args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)



if args.dataset == "MNIST":
    net = MNIST()
    net = torch.nn.DataParallel(net, device_ids=[0])
    load_model(net,'model/mnist_gpu.pt')
    train_loader, test_loader, train_dataset, test_dataset = load_mnist_data(args.test_batch_size)
elif args.dataset == 'CIFAR10':
    net = CIFAR10() 
    #net = VGG_plain('VGG16', 10, img_width=32)
    net = torch.nn.DataParallel(net, device_ids=[0])
    load_model(net,args.model_dir)

    #device = torch.device("cuda")
    #net = WideResNet().to(device)
    #load_model(net, 'model/cifar10_gpu.pt')

    train_loader, test_loader, train_dataset, test_dataset = load_cifar10_data(args.test_batch_size)
elif args.dataset == 'Imagenet':
    net = CIFAR10() 
    net = torch.nn.DataParallel(net, device_ids=[0])
    load_model(net, 'cifar10_gpu.pt')
else:
    print("Unsupport dataset")
    #os.exit(0)

attack_list = {
    "PGD":PGD,
    "Sign_OPT": OPT_attack_sign_SGD,
    "Sign_OPT_lf": OPT_attack_sign_SGD_lf,
    "CW": CW,
    "OPT_attack": OPT_attack,
    "HSJA": HSJA,
    "OPT_attack_lf": OPT_attack_lf,
    "FGSM": FGSM,
    "NES": NES,
    "Bandit": Bandit,
    "NATTACK": NATTACK,
    "Sign_SGD": Sign_SGD,
    "ZOO": ZOO,
    "Liu": OPT_attack_sign_SGD_v2,
    "Evolutionary": Evolutionary
}


net.cuda()
net.eval()

model = net 


amodel = PytorchModel(model, bounds=[0,1], num_classes=10)
if args.attack=="Bandit":
    attack = attack_list[args.attack](amodel,args.exploration,args.fd_eta,args.online_lr,args.mode)
else:
    attack = attack_list[args.attack](amodel)



total_r_count = 0
total_clean_count = 0

#logs = torch.zeros(1000,2)

for i, (xi,yi) in enumerate(test_loader):
    print(f"image batch: {i}")
    if i==args.test_batch:
        #continue
        break
    xi,yi = xi.cuda(), yi.cuda()
    #adv=attack(xi,yi, 0.2)
    adv=attack(xi,yi,TARGETED=args.targeted)

    if args.targeted == False:
        r_count= (torch.max(amodel.predict(adv),1)[1]==yi).nonzero().shape[0]
        clean_count= (torch.max(amodel.predict(xi),1)[1]==yi).nonzero().shape[0]
        total_r_count += r_count
        total_clean_count += clean_count
    if args.attack in ["Sign_OPT","OPT_attack"]:
        if i==0:
            logs = torch.zeros(attack.get_log().size())
        logs += attack.get_log()

if args.attack in ["Sign_OPT","OPT_attack"]:
    logs /= args.test_batch
    print("saving logs to numpy array")
    np.save("attack_log.npy",logs.numpy())
    import matplotlib.pyplot as plt
    plot_log = np.load("attack_log.npy")
    plt.plot(plot_log[:,1],plot_log[:,0])
    plt.ylabel('Distortion')
    plt.xlabel('Num of queries')
    plt.show()
    plt.savefig('attack_plot.png')

else:
    num_queries = amodel.get_num_queries()
    #print(i, total_r_count, total_clean_count)
    print(f"clean count:{total_clean_count}")
    print(f"acc under attack count:{total_r_count}")
    print(f"number queries used:{num_queries}")
    #print(clean_count - r_count, (clean_count - r_count)/clean_count)
