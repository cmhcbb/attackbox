import torch
from torch.autograd import Variable
import torch.optim as optim
from utils import mulvt

class CWattack(object):
    def __init__(self,model):
        self.model = model

    def get_loss(self,xi,label_or_target,TARGETED):
        criterion = nn.CrossEntropyLoss()
        output = self.model.predict(xi)
        loss = criterion(output, label_or_target)
        #print(c.size(),modifier.size())
        return loss

    def fgsm(self, input_xi, label_or_target, c, TARGETED=False):
       
        yi = Variable(label_or_target.cuda())
        xi = Variable(input_xi.cuda())
        for it in range(10):
            xi.gradient.zero_()
            error = self.get_loss(xi,yi,TARGETED)
            self.model.get_gradient(error)
            gradient = xi.grad
            xi = xi - eta* torch.sign(gradient)
            #error.backward()
            if (it)%1==0:
                print(loss.data[0]) 
        return xi
 

    def __call__(self, input_xi, label_or_target, eta, TARGETED=False):
        adv = self.fgsm(input_xi, label_or_target, eta, TARGETED)
        return adv   
    
        
