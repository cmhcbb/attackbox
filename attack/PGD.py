from __future__ import division
from __future__ import print_function

import time
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
class PGD(object):
    def __init__(self,model):
        self.model = model
    
    def random_start(self,x,eps):
        x+=torch.FloatTensor(x.size()).uniform_(-eps,eps).cuda()
        x.clamp_(0,1)
        return x

    def get_loss(self,xi,label_or_target,TARGETED):
        criterion = nn.CrossEntropyLoss()
        output = self.model.predict(xi)
        #print(output.shape)
        #print(output, label_or_target)
        loss = criterion(output, label_or_target)
        #print(loss)
        #print(c.size(),modifier.size())
        return loss
    
    def pgd(self, input_xi, label_or_target, epsilon, eta, TARGETED=False, random_start=True):
        #yi = Variable(label_or_target)
        #x_adv = Variable(input_xi.cuda(), requires_grad=True)
        yi = label_or_target
        x_adv = input_xi.clone()
        if random_start:
            x_adv = self.random_start(x_adv,epsilon)
        x_adv.requires_grad = True
        for it in range(100):
            error = self.get_loss(x_adv,yi, TARGETED)
            #if (it)%10==0:
            #    print(error.data.item()) 
            #x_adv.grad.data.zero_()
            #error.backward(retain_graph=True)
            #print(error.requires_grad)
            self.model.get_gradient(error)
            x_adv.grad.sign_()
            if TARGETED:
                x_adv.data = x_adv.data - eta* epsilon * x_adv.grad.data
            else:
                x_adv.data = x_adv.data + eta* epsilon * x_adv.grad.data
            diff = x_adv.data - input_xi
            diff.clamp_(-epsilon,epsilon)
            x_adv.data=(diff + input_xi).clamp_(0, 1)
            x_adv.grad.data.zero_()
        return x_adv

    def __call__(self, input_xi, label_or_target, epsilon=0.01, eta=0.5, TARGETED=False):
        adv = self.pgd(input_xi, label_or_target, epsilon, eta, TARGETED)
        return adv  

