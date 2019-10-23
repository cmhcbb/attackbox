import time
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.modules import Upsample

def norm(t):
    assert len(t.shape) == 4
    norm_vec = torch.sqrt(t.pow(2).sum(dim=[1,2,3])).view(-1, 1, 1, 1)
    norm_vec += (norm_vec == 0).float()*1e-8
    return norm_vec

###
# Different optimization steps
# All take the form of func(x, g, lr)
# eg: exponentiated gradients
# l2/linf: projected gradient descent
###

def eg_step(x, g, lr):
    real_x = (x + 1)/2 # from [-1, 1] to [0, 1]
    pos = real_x*torch.exp(lr*g)
    neg = (1-real_x)*torch.exp(-lr*g)
    new_x = pos/(pos+neg)
    return new_x*2-1

def linf_step(x, g, lr):
    return x + lr*torch.sign(g)

def l2_prior_step(x, g, lr):
    new_x = x + lr*g/norm(g)
    norm_new_x = norm(new_x)
    norm_mask = (norm_new_x < 1.0).float()
    return new_x*norm_mask + (1-norm_mask)*new_x/norm_new_x

def gd_prior_step(x, g, lr):
    return x + lr*g
   
def l2_image_step(x, g, lr):
    return x + lr*g/norm(g)

##
# Projection steps for l2 and linf constraints:
# All take the form of func(new_x, old_x, epsilon)
##

def l2_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        delta = new_x - orig
        out_of_bounds_mask = (norm(delta) > eps).float()
        x = (orig + eps*delta/norm(delta))*out_of_bounds_mask
        x += new_x*(1-out_of_bounds_mask)
        return x
    return proj

def linf_proj(image, eps):
    orig = image.clone()
    def proj(new_x):
        return orig + torch.clamp(new_x - orig, -eps, eps)
    return proj


class Bandit(object):
    def __init__(self,model,exploration,fd_eta,online_lr,mode):
        self.model = model
        self.exploration = exploration
        self.fd_eta = fd_eta
        self.online_lr = online_lr
        self.mode = mode

    def get_loss(self, xi, label_or_target):
        criterion = nn.CrossEntropyLoss()
        output = self.model.predict_prob(xi)
        loss = criterion(output, label_or_target)
        return loss


    def bandit(self,input_xi, label, epsilon, eta, TARGETED):
        img_dim = input_xi.size(-1)
        #channel = input_xi.size(0)
        batch_size = input_xi.size(0)
        upsampler = Upsample(size=(img_dim,img_dim))
        prior = torch.zeros(input_xi.size()).cuda()
        dim = prior.numel()/batch_size
        prior_step = gd_prior_step if self.mode == 'l2' else eg_step
        image_step = l2_image_step if self.mode == 'l2' else linf_step
        proj_maker = l2_proj if self.mode == 'l2' else linf_proj
        image = input_xi.detach().clone()
        print(image.max(), image.min())
        proj_step = proj_maker(image, epsilon)
        
        orig_classes = self.model.predict_prob(input_xi).argmax(1).cuda()
        correct_classified_mask = (orig_classes == label).float()
 
        
        for _ in range(1000):
            exp_noise = self.exploration*torch.randn_like(prior)/(dim**0.5)
            exp_noise = exp_noise.cuda()
            # Query deltas for finite difference estimator
            q1 = upsampler(prior + exp_noise).cuda()
            q2 = upsampler(prior - exp_noise).cuda()
            # Loss points for finite difference estimator
            #print(q1.size())
            l1 = self.get_loss(image+self.fd_eta*q1/norm(q1),label)
            l2 = self.get_loss(image+self.fd_eta*q2/norm(q2),label)
            #print(l1.data.item())
            #l1 = L(image + args.fd_eta*q1/norm(q1)) # L(prior + c*noise)
            #l2 = L(image + args.fd_eta*q2/norm(q2)) # L(prior - c*noise)
            # Finite differences estimate of directional derivative
            est_deriv = (l1 - l2)/(self.fd_eta*self.exploration)
            # 2-query gradient estimate
            est_grad = est_deriv.view(-1, 1, 1, 1)*exp_noise
            # Update the prior with the estimated gradient
            prior = prior_step(prior, est_grad, self.online_lr)
            new_im = image_step(image, upsampler(prior*correct_classified_mask.view(-1, 1, 1, 1)), eta)
            image = proj_step(new_im)
            image = torch.clamp(image, 0, 1)
        
        return image 
    
    def __call__(self, input_xi, label_or_target, epsilon=0.01, eta=0.01, TARGETED=False):
        adv = self.bandit(input_xi, label_or_target, epsilon, eta, TARGETED)
        return adv  
