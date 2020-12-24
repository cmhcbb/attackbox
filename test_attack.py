import torch
#from wideresnet import *
import os, argparse, logging, sys, shutil
import numpy as np
import utils
import shutil
import matplotlib.pyplot as plt
from attack import *
from models import PytorchModel
from paper_model import vgg16, BasicCNN
from allmodels import MNIST, load_model, load_mnist_data, load_cifar10_data, CIFAR10, VGG_plain, VGG_rse, VGG_vi
from robustbench.utils import load_model as load_model_aa


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
parser.add_argument('--model', type=str, required=True, help='model to be attacked')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--query', type=int, help='Query limit allowed')
parser.add_argument('--save', type=str, default='', help='exp_id')
parser.add_argument('--exp_tag', type=str, default='')
parser.add_argument('--gpu', type=str, default='auto', help='tag for saving, enter debug mode if debug is in it')

args = parser.parse_args()


#### env
np.random.seed(args.seed)
torch.manual_seed(args.seed)
gpu = utils.pick_gpu_lowest_memory() if args.gpu == 'auto' else int(args.gpu)
torch.cuda.set_device(gpu)
print('gpu:', gpu)


#### macros
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
linf_list = ["PGD","Sign_OPT_lf","OPT_attack_lf","NES", "Sign_OPT_lf_bvls"]

if args.attack in l2_list:
    norm = 'L2'
elif args.attack in linf_list:
    norm = 'Linf'


#### dir managemet
exp_id = args.save
args.save = './experiments/{}-{}'\
    .format(exp_id, args.model)
if args.exp_tag != '':
    args.save += '-{}'.format(args.exp_tag)

scripts_to_save = ['./exp_scripts/{}'.format(exp_id + '.sh')]
if os.path.exists(args.save):
    if 'debug' in args.exp_tag or input("WARNING: {} exists, override?[y/n]".format(args.save)) == 'y':
        shutil.rmtree(args.save)
    else: exit()
utils.create_exp_dir(args.save, scripts_to_save=scripts_to_save)


#### logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
log_file = 'log.txt'
log_path = os.path.join(args.save, log_file)
logging.info('======> log filename: %s', log_file)
if os.path.exists(log_path):
    if input("WARNING: {} exists, override?[y/n]".format(log_file)) == 'y':
        print('proceed to override log file directory')
    else: exit()
fh = logging.FileHandler(log_path, mode='w')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


#### load data
if args.dataset == "MNIST":
    train_loader, test_loader, train_dataset, test_dataset = load_mnist_data(args.test_batch_size)
elif args.dataset == 'CIFAR10':
    train_loader, test_loader, train_dataset, test_dataset = load_cifar10_data(args.test_batch_size)
elif args.dataset == 'Imagenet':
    print('unsupported right now')
    exit(1)
else:
    print("Unsupport dataset")

logging.info(args)

#### load model
## load defense model
## clean
aa_model_dir = './model/defense_models'
if   args.model == 'mnist':
    # model = MNIST()
    model = BasicCNN()
    model = torch.nn.DataParallel(model, device_ids=[gpu])
    load_model(model, 'model/mnist_gpu.pt')
elif args.model == 'cifar10':
    # model = vgg16()
    # model = VGG_plain('VGG16', 10, img_width=32)
    #model = WideResmodel().to(device)
    model = CIFAR10()
    model = torch.nn.DataParallel(model, device_ids=[gpu])
    load_model(model, 'model/cifar10_gpu.pt')
## linf
elif args.model == 'Sehwag2020Hydra' or args.model == 'hydra': # Hydra 
    model = load_model_aa(model_name='Sehwag2020Hydra', model_dir=aa_model_dir, norm=norm)
elif args.model == 'Wang2020Improving' or args.model == 'mart': # 
    model = load_model_aa(model_name='Wang2020Improving', model_dir=aa_model_dir, norm=norm)
elif args.model == 'Zhang2019Theoretically' or args.model == 'trades': # TRADES
    model = load_model_aa(model_name='Zhang2019Theoretically', model_dir=aa_model_dir, norm=norm)
elif args.model == 'Wong2020Fast' or args.model == 'fastat': # Fast AT
    model = load_model_aa(model_name='Wong2020Fast', model_dir=aa_model_dir, norm=norm)
## l2
elif args.model == 'Wu2020Adversarial':
    model = load_model_aa(model_name='Wu2020Adversarial', model_dir=aa_model_dir, norm=norm)
elif args.model == 'Augustin2020Adversarial':
    model = load_model_aa(model_name='Augustin2020Adversarial', model_dir=aa_model_dir, norm=norm)
elif args.model == 'Rice2020Overfitting':
    model = load_model_aa(model_name='Rice2020Overfitting', model_dir=aa_model_dir, norm=norm)
else:
    print('unsupported model'); exit(1)

model.cuda()
model.eval()

## load attack model
# sign opt
amodel = PytorchModel(model, bounds=[0,1], num_classes=10) # just a wrapper
if args.attack=="Bandit":
    attack = attack_list[args.attack](amodel,args.exploration,args.fd_eta,args.online_lr,args.mode)
else:
    attack = attack_list[args.attack](amodel)


total_r_count = 0
total_clean_count = 0
total_distance = 0
rays_successes = []
successes = []
stop_queries = [] # wrc added to match RayS
for i, (xi, yi) in enumerate(test_loader):
    logging.info(f"image batch: {i}")
    
    ## data
    if i == args.test_batch: break
    xi, yi = xi.cuda(), yi.cuda()
    
    ## attack
    theta_init = None
    adv, distortion, is_success, nqueries, theta_signopt = attack(xi, yi,
        targeted=args.targeted, query_limit=args.query, distortion=args.epsilon, args=args)

    if theta_init is not None:
        match = (theta_signopt.astype(np.int32) == theta_init.astype(np.int32)).sum() / np.sum(abs(theta_signopt))
        print('sign matching rate between theta_init and theta_signopt:', match)

    if is_success:
        stop_queries.append(nqueries)

    if args.targeted == False:
        r_count = (torch.max(amodel.predict(adv),1)[1]==yi).nonzero().shape[0]
        clean_count = (torch.max(amodel.predict(xi),1)[1]==yi).nonzero().shape[0]
        total_r_count += r_count
        total_clean_count += clean_count
        total_distance += utils.distance(adv,xi,norm=norm.lower())
    # if args.attack in ["Sign_OPT","OPT_attack"]:
    #     if i == 0:
    #         logs = torch.zeros(attack.get_log().size())
    #     logs += attack.get_log()
    successes.append(is_success)


# if args.attack in ["Sign_OPT", "OPT_attack"]:
#     logs /= args.test_batch
#     logging.info("saving logs to numpy array")
#     npy_file = args.dataset + args.attack + "_log.npy"
#     npy_file_path = os.path.join(args.save, npy_file)
#     np.save(npy_file_path, logs.numpy())
#     plot_log = np.load(npy_file)
#     plt.plot(plot_log[:,1],plot_log[:,0])
#     plt.ylabel('Distortion')
#     plt.xlabel('Num of queries')
#     plt.show()
#     png_file = args.dataset + args.attack + "_plot.png"
#     png_file_path = os.path.join(args.save, png_file)
#     plt.savefig(png_file_path)
# else:
num_queries = amodel.get_num_queries()
#logging.info(i, total_r_count, total_clean_count)
logging.info("="*10)
logging.info(f"clean count:{total_clean_count}")
logging.info(f"acc under attack count:{total_r_count}")
logging.info(f"avg total queries used:{num_queries}")
logging.info(f"avg stop queries used:{np.mean(stop_queries)}")
logging.info(f"average distortion:{total_distance/(args.test_batch*args.test_batch_size)}")
#logging.info(clean_count - r_count, (clean_count - r_count)/clean_count)
logging.info("robust accuracy rays: {}".format(1 - np.mean(rays_successes)))
logging.info("robust accuracy: {}".format(1 - np.mean(successes)))