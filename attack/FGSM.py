import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

class FGSM(object):
    def __init__(self,model):
        self.model = model

    def get_loss(self,xi,label_or_target,TARGETED):
        criterion = nn.CrossEntropyLoss()
        output = self.model.predict(xi)
        #print(output, label_or_target)
        loss = criterion(output, label_or_target)
        #print(loss)
        #print(c.size(),modifier.size())
        return loss

    def i_fgsm(self, input_xi, label_or_target, eta, TARGETED=False):
       
        yi = Variable(label_or_target.cuda())
        x_adv = Variable(input_xi.cuda(), requires_grad=True)
        for it in range(10):
            error = self.get_loss(x_adv,yi,TARGETED)
            if (it)%1==0:
                print(error.item()) 
            self.model.get_gradient(error)
            #print(gradient)
            x_adv.grad.sign_()
            if TARGETED:
                x_adv.data = x_adv.data - eta* x_adv.grad 
            else:
                x_adv.data = x_adv.data + eta* x_adv.grad
            #x_adv = Variable(x_adv.data, requires_grad=True)
            #error.backward()
        return x_adv

    def fgsm(self, input_xi, label_or_target, eta, TARGETED=False):
       
        yi = Variable(label_or_target.cuda())
        x_adv = Variable(input_xi.cuda(), requires_grad=True)

        error = self.get_loss(x_adv,yi,TARGETED)
        print(error.item()) 
        self.model.get_gradient(error)
        #print(gradient)
        x_adv.grad.sign_()
        if TARGETED:
            x_adv.data = x_adv.data - eta* x_adv.grad 
        else:
            x_adv.data = x_adv.data + eta* x_adv.grad
            #x_adv = Variable(x_adv.data, requires_grad=True)
            #error.backward()
        return x_adv 

    def __call__(self, input_xi, label_or_target, eta=0.01, TARGETED=False, ITERATIVE=False, epsilon=None):
        if ITERATIVE:
            adv = self.i_fgsm(input_xi, label_or_target, eta, TARGETED)
        else:
            eta = epsilon
            adv = self.fgsm(input_xi, label_or_target, eta, TARGETED)
        return adv   
    
        
