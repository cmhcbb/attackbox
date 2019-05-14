import argparse
import os
import sys

from OPT_attack import OPT_attack
from OPT_attack_lf import OPT_attack_lf
from OPT_attack_sign_SGD import OPT_attack_sign_SGD
from OPT_attack_sign_SGD_lf import OPT_attack_sign_SGD_lf
from models import PytorchModel
import torch
import torchvision.models as models
from allmodels import  load_model, MNIST, load_mnist_data, load_cifar10_data, CIFAR10, IMAGENET, load_imagenet_data
import numpy as np
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser(description='BlackBox attack - Sign SGD')

parser.add_argument('algorithm', default='sign_sgd', type=str,
                    help='Algorithm for attack -> sign_sgd | opt')
parser.add_argument('dataset', default='mnist', type=str,
                    help='Dataset -> mnist | cifar | imagenet')
parser.add_argument('--targeted', default='false', type=str,
                    help='Whether targeted or untargeted')
parser.add_argument('--norm', default='l2', type=str,
                    help='Norm for attack -> l2 | linf')
parser.add_argument('--num', default=50, type=int,
                    help='Number of samples to be attacked from test dataset.')
parser.add_argument('--stop', default=None, type=float,
                    help='Stopping threshold')
parser.add_argument('--query_limit', default=40000, type=int,
                    help='Maximum queries for the attack')
parser.add_argument('--start_from', default=0, type=int,
                    help='Number of samples to be skipped before attack.')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

def attack(algorithm, dataset, targeted, norm='l2', num=50, stopping_criteria=None,
           query_limit=40000, start_from=0, gpu=0):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu);
    
    print("Attacking:".format(num))
    print("    Number of samples - {0}".format(num))
    print("    Dataset - {0}".format(dataset.upper()))
    print("    Targeted - {0}".format(targeted))
    print("    Norm - {0}".format(norm))
    print("    Query Limit - {0}".format(query_limit))
    print("    GPU - {0}".format(gpu))
    print()
    if stopping_criteria is not None:
        print("    Stopping criteria - {0}".format(stopping_criteria))
    if start_from > 0:
        print("    Start from {0}".format(start_from))
    
    if dataset == 'mnist':
        net = MNIST()
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        load_model(net,'mnist_gpu.pt')
        train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()
    elif dataset == 'cifar':
        net = CIFAR10()
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        load_model(net,'cifar10_gpu.pt')
        train_loader, test_loader, train_dataset, test_dataset = load_cifar10_data()
    elif dataset == 'imagenet':
        net = models.__dict__["resnet50"](pretrained=True)
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        train_loader, test_loader, train_dataset, test_dataset = load_imagenet_data()
    else:
        print("Invalid dataset")
        return

    net.eval()
    model = net.module if torch.cuda.is_available() else net
    amodel = PytorchModel(model, bounds=[0,1], num_classes=10)
    
    attack_type = None
    if algorithm == 'opt':
        if norm=='l2':
            attack_type = OPT_attack
        elif norm=='linf':
            attack_type = OPT_attack_lf
    elif algorithm == 'sign_sgd':
        if norm=='l2':
            attack_type = OPT_attack_sign_SGD
        elif norm=='linf':
            attack_type = OPT_attack_sign_SGD_lf    
    else:
        print("Invalid algorithm")
        
    if attack_type is None:
        print("Invalid norm")
    
    if targeted:
        attack = attack_type(amodel, train_dataset=train_dataset)
    else:
        attack = attack_type(amodel)
        
    np.random.seed(0)
    seeds = np.random.randint(10000, size=[2*num])
    count = 0
    for i, (xi,yi) in enumerate(test_loader):
        if i < start_from:
            continue
        if count == num:
            break

        seed_index = i - start_from
        np.random.seed(seeds[seed_index])
        target = np.random.randint(10)*torch.ones(1, dtype=torch.long).cuda() if targeted else None
        print("Attacking Source: {0} Target: {1} Seed: {2} Number {3}".format(yi.item(), target, seeds[seed_index], i))
        adv, dist = attack(xi.cuda(), yi.cuda(), target=target,
                           seed=seeds[seed_index], query_limit=query_limit)
        if dist > 1e-8 and dist != float('inf'):
            count += 1
        print()

if __name__ == "__main__":
    args = parser.parse_args()
    if args.targeted.lower() in ['yes', 'true', 't', 'y', '1']:
        targeted = True
    elif args.targeted.lower() in ['no', 'false', 'f', 'n', '0']:
        targeted = False
    else:
        print("Error: targeted should be true/false | t/f | y/n | 0/1")
        exit(1)
    attack(args.algorithm, args.dataset, targeted, norm=args.norm, num=args.num, stopping_criteria=args.stop,
           query_limit=args.query_limit, start_from=args.start_from, gpu=args.gpu)