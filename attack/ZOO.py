import torch
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

class ZOO(object):
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
        step_size = 0.1
        modifier = Variable(torch.zeros(input_xi.size()).cuda())
        yi = label_or_target
        label_onehot = torch.FloatTensor(yi.size()[0],self.model.num_classes)
        label_onehot.zero_()
        label_onehot.scatter_(1,yi.view(-1,1),1)
        label_onehot_v = Variable(label_onehot, requires_grad=False).cuda()
        xi = Variable(input_xi.cuda())
        #optimizer = optim.Adam([modifier], lr = 0.1)
        best_loss1 = 1000
        best_adv = None
        num_coor = 1
        delta = 0.0001
        for it in range(20000):
            #optimizer.zero_grad()
            error1,loss11,loss12 = self.get_loss(xi,label_onehot_v,c,modifier, TARGETED)
            for j in range(num_coor):
                modifier = Variable(torch.zeros(xi.size()).cuda(), volatile=True)
                randx = np.random.randint(xi.size()[0])
                randy = np.random.randint(xi.size()[1])
                randz = np.random.randint(xi.size()[2])
                modifier[randx,randy,randz] = delta
                #print(modifier)
                new_xi = xi + modifier
                error2,loss21,loss22 = self.get_loss(new_xi,label_onehot_v,c,modifier, TARGETED)
                modifier_gradient = (error2 - error1) / delta * modifier
                modifier -= step_size*modifier_gradient
            xi = xi + modifier
            #self.model.get_gradient(error)
            #error.backward()
            #optimizer.step()
            if (it)%1000==0:
                print(error1.data[0],loss11.data[0],loss12.data[0]) 
        return xi
    
    def random_zoo(self, input_xi, label_or_target, c, TARGETED=False):
        step_size = 5e-3    
        modifier = Variable(torch.zeros(input_xi.size()).cuda())
        yi = label_or_target
        label_onehot = torch.FloatTensor(yi.size()[0],self.model.num_classes)
        label_onehot.zero_()
        label_onehot.scatter_(1,yi.view(-1,1),1)
        label_onehot_v = Variable(label_onehot, requires_grad=False).cuda()
        xi = Variable(input_xi.cuda(),requires_grad=False)
        #optimizer = optim.Adam([modifier], lr = 0.1)
        best_loss1 = 1000
        best_adv = None
        num_coor = 1
        delta = 1e-6
        modifier = Variable(torch.zeros(xi.size()).cuda(), volatile=True)
        for it in range(20000):
            #optimizer.zero_grad()
            error1,loss11,loss12 = self.get_loss(xi,label_onehot_v,c,modifier, TARGETED)
            u=torch.randn(xi.size()).cuda()
            error2,loss21,loss22 = self.get_loss(xi,label_onehot_v,c,modifier+delta*u, TARGETED)
            modifier_gradient = (error2 - error1) / delta * u
            modifier.data = modifier.data - step_size*modifier_gradient
            #xi = xi + modifier
            #self.model.get_gradient(error)
            #error.backward()
            #optimizer.step()
            if (it)%100==0:
                print(it,error1.item(),loss11.item(),loss12.item()) 
        return xi        

    def __call__(self, input_xi, label_or_target, c=0.1, TARGETED=False):
        adv = self.random_zoo(input_xi, label_or_target, c, TARGETED)
        return adv   
    
        
