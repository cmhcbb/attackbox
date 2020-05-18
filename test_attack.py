from attack import *
from models import PytorchModel
import torch
from allmodels import MNIST, load_model, load_mnist_data, load_cifar10_data, CIFAR10, VGG_plain, VGG_rse, VGG_vi
#from wideresnet import *
from paper_model import vgg16, BasicCNN
import os, argparse
import numpy as np
import utils

parser = argparse.ArgumentParser()
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
parser.add_argument('--epsilon', type=float, default=0.01,
                        help='epsilon in the PGD attack')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='verbose.')
parser.add_argument('--test_batch_size', type=int, default=1,
                    help='test batch_size')
parser.add_argument('--test_batch', type=int, default=10,
                    help='test batch number')
parser.add_argument('--model_dir', type=str, required=True, help='model loading directory')
parser.add_argument('--seed', type=int, default=1, help='random seed')

args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)



if args.dataset == "MNIST":
    # net = MNIST()
    net = BasicCNN()
    # net = torch.nn.DataParallel(net, device_ids=[0])
    load_model(net,args.model_dir)
    net = torch.nn.DataParallel(net, device_ids=[0])
    train_loader, test_loader, train_dataset, test_dataset = load_mnist_data(args.test_batch_size)
elif args.dataset == 'CIFAR10':
    # net = CIFAR10() 
    net = vgg16()
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
    load_model(net,args.model_dir)
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
    "Evolutionary": Evolutionary,
    "SimBA": SimBA
}

l2_list = ["Sign_OPT","CW", "OPT_attack","FGSM","ZOO","SimBA"]
linf_list = ["PGD","Sign_OPT_lf","OPT_attack_lf","NES"]

if args.attack in l2_list:
    norm = 'l2'
elif args.attack in linf_list:
    norm = 'linf'


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
total_distance = 0
#logs = torch.zeros(1000,2)

for i, (xi,yi) in enumerate(test_loader):
    print(f"image batch: {i}")
    if i == args.test_batch:
        # continue
        break
    xi,yi = xi.cuda(), yi.cuda()
    #adv=attack(xi,yi, 0.2)
    adv=attack(xi,yi,epsilon=args.epsilon, TARGETED=args.targeted)

    if args.targeted == False:
        r_count= (torch.max(amodel.predict(adv),1)[1]==yi).nonzero().shape[0]
        clean_count= (torch.max(amodel.predict(xi),1)[1]==yi).nonzero().shape[0]
        total_r_count += r_count
        total_clean_count += clean_count
        total_distance += utils.distance(adv,xi,norm=norm)
    if args.attack in ["Sign_OPT","OPT_attack"]:
        if i==0:
            logs = torch.zeros(attack.get_log().size())
        logs += attack.get_log()

if args.attack in ["Sign_OPT","OPT_attack"]:
    logs /= args.test_batch
    print("saving logs to numpy array")
    npy_file = args.dataset + args.attack + "_log.npy"
    np.save(npy_file,logs.numpy())
    import matplotlib.pyplot as plt
    plot_log = np.load(npy_file)
    plt.plot(plot_log[:,1],plot_log[:,0])
    plt.ylabel('Distortion')
    plt.xlabel('Num of queries')
    plt.show()
    png_file = args.dataset + args.attack + "_plot.png"
    plt.savefig(png_file)

else:
    num_queries = amodel.get_num_queries()
    #print(i, total_r_count, total_clean_count)
    print(f"clean count:{total_clean_count}")
    print(f"acc under attack count:{total_r_count}")
    print(f"number queries used:{num_queries}")
    print(f"average distortion:{total_distance/(args.test_batch*args.test_batch_size)}")
    #print(clean_count - r_count, (clean_count - r_count)/clean_count)
