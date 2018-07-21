import torch
from torch.autograd import Variable
import torch.optim as optim
from utils import mulvt
import numpy as np

class CWattack(object):
    def __init__(self,model):
        self.model = model
        
    def get_loss(self,xi,label_onehot_v, c, modifier, TARGETED):
        #print(c.size(),modifier.size())
        loss1 = c*torch.sum(modifier*modifier)
        #output = net(torch.clamp(xi+modifier,0,1))
        output = self.model.predict(xi+modifier)
        real = torch.max(torch.mul(output, label_onehot_v), 1)[0]
        other = torch.max(torch.mul(output, (1-label_onehot_v))-label_onehot_v*10000,1)[0]
        #print(real,other)
        if TARGETED:
            loss2 = torch.sum(torch.clamp(other - real, min=0))
        else:
            loss2 = torch.sum(torch.clamp(real - other, min=0))
        error = loss2 + loss1 
        return error,loss1,loss2

    def zoo(self, input_xi, label_or_target, c, TARGETED=False):
       
        modifier = Variable(torch.zeros(input_xi.size()).cuda(), requires_grad=True)
        yi = label_or_target
        label_onehot = torch.FloatTensor(yi.size()[0],self.model.num_classes)
        label_onehot.zero_()
        label_onehot.scatter_(1,yi.view(-1,1),1)
        label_onehot_v = Variable(label_onehot, requires_grad=False).cuda()
        xi = Variable(input_xi.cuda())
        #optimizer = optim.Adam([modifier], lr = 0.1)
        best_loss1 = 1000
        best_adv = None
        num_coor =5
        delta = 0.0001
        for it in range(1000):
            #optimizer.zero_grad()
            error1,loss11,loss12 = self.get_loss(xi,label_onehot_v,c,modifier, TARGETED)
            modifier = Variable(torch.zeros(xi.size()))
            for j in range(num_coor):
                randx = np.random.randint(xi.size()[0])
                randy = np.random.randint(xi.size()[1])
                randz = np.random.randint(xi.size()[2])
                modifier[randx][randy][randz] = delta
                new_xi = xi + modifier
                error2,loss21,loss22 = self.get_loss(new_xi,label_onehot_v,c,modifier, TARGETED)
                modifier_gradient = (error2 - error1) / delta * modifier
                modifier -= step_size*modifier_gradient
            xi = xi + modifier
            #self.model.get_gradient(error)
            #error.backward()
            #optimizer.step()
            if (it)%500==0:
                print(error.data[0],loss1.data[0],loss2.data[0]) 
            return xi
 

    def __call__(self, input_xi, label_or_target, TARGETED=False):
        adv = self.zoo(input_xi, label_or_target, c_v, TARGETED)
        return adv   
    
        
