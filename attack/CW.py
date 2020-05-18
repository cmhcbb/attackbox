import torch
from torch.autograd import Variable
import torch.optim as optim

class CW(object):
    def __init__(self,model):
        self.model = model

    def get_loss(self,xi,label_onehot_v, c, modifier, TARGETED):
        #print(c.size(),modifier.size())
        loss1 = c*torch.sum(modifier*modifier)
        #output = net(torch.clamp(xi+modifier,0,1))
        output = self.model.predict(xi+modifier)
        real = torch.sum(torch.mul(output, label_onehot_v), 1)
        other = torch.max(torch.mul(output, (1-label_onehot_v))-label_onehot_v*10000,1)[0]
        #print(real,other)
        if TARGETED:
            loss2 = torch.sum(torch.clamp(other - real, min=0))
        else:
            loss2 = torch.sum(torch.clamp(real - other, min=0))
        error = loss2 + loss1.sum() 
        #print(loss1,loss2)
        return error,loss1,loss2

    def cw(self, input_xi, label_or_target, c, TARGETED=False):
       
        modifier = Variable(torch.zeros(input_xi.size()).cuda(), requires_grad=True)
        yi = label_or_target
        label_onehot = torch.FloatTensor(yi.size()[0],self.model.num_classes).cuda()
        label_onehot.zero_()
        label_onehot.scatter_(1,yi.view(-1,1),1)
        label_onehot_v = Variable(label_onehot, requires_grad=False).cuda()
        xi = Variable(input_xi.cuda())
        optimizer = optim.Adam([modifier], lr = 0.1)
        best_loss1 = 1000
        best_adv = None
        for it in range(1000):
            optimizer.zero_grad()
            error,loss1,loss2 = self.get_loss(xi,label_onehot_v,c,modifier, TARGETED)
            #print(error)
            self.model.get_gradient(error)
            #error.backward()
            #if (it)%500==0:
            #    print(error.data,loss1.data,loss2.data) 
            if loss2.data ==0:
                if best_loss1 >= loss1.data:
                    best_loss1 = loss1.data
                    best_adv = modifier.clone()    
            optimizer.step()            # if (it)%10==0:
            #    print(error.data.item()) 
        if best_adv is None:
            #print(str(c)+'\t'+'None')
            return None
        else:
            return best_adv
 

    def __call__(self, input_xi, label_or_target, epsilon=None, TARGETED=False):
        dis_a = []
        c_hi = 1000*torch.ones(input_xi.size()[0],1).cuda()
        c_lo = 0.01*torch.ones(c_hi.size()).cuda()
        while torch.max(c_hi-c_lo) > 1e-1:
            c_mid = (c_hi + c_lo)/2.0
            adv = self.cw(input_xi, label_or_target, c_mid, TARGETED)
            if adv is None:
                c_hi = c_mid
                #print(c_mid)
            else:
                dis = torch.norm(adv).data
                dis_a.append(dis)
                print(dis)
                c_lo = c_mid
        return adv   
    
        
