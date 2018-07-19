import torch
from torch.autograd import Variable
import torch.optim as optim
from utils import mulvt

class CWattack(object):
    def __init__(self,model):
        self.model = model

    def get_loss(self,xi,label_onehot_v, c, modifier, TARGETED):
        #print(c.size(),modifier.size())
        #loss1 = c*torch.sum(modifier*modifier)
        newt = mulvt(c,modifier)
        loss1 = (newt*newt).sum(3).sum(2).sum(1)
        #output = net(torch.clamp(xi+modifier,0,1))
        output = self.model.predict(xi+modifier)
        real = torch.max(torch.mul(output, label_onehot_v), 1)[0]
        other = torch.max(torch.mul(output, (1-label_onehot_v))-label_onehot_v*10000,1)[0]
        #print(real,other)
        if TARGETED:
            loss2 = torch.clamp(other - real, min=0).sum(3).sum(2).sum(1)
        else:
            loss2 = torch.clamp(real - other, min=0).sum(3).sum(2).sum(1)
        error = loss2 + loss1 
        return error,loss1,loss2

    def cw(self, input_xi, label_or_target, c, TARGETED=False):
        batch_size = input_xi.size()[0] 
        modifier = Variable(torch.zeros(input_xi.size()).cuda(), requires_grad=True)
        yi = label_or_target
        label_onehot = torch.FloatTensor(yi.size()[0],self.model.num_classes)
        label_onehot.zero_()
        label_onehot.scatter_(1,yi.view(-1,1),1)
        label_onehot_v = Variable(label_onehot, requires_grad=False).cuda()
        xi = Variable(input_xi.cuda())
        optimizer = optim.Adam([modifier], lr = 0.1)
        best_loss1 = 1000 * torch.ones(batch_size,1)
        #best_adv = None
        for it in range(1000):
            optimizer.zero_grad()
            error,loss1,loss2 = self.get_loss(xi,label_onehot_v,c,modifier, TARGETED)
            s_error = torch.sum(error)
            self.model.get_gradient(s_error)
            #error.backward()
            optimizer.step()
            if (it)%500==0:
                print(s_error.data[0])
            
            candidate=(loss2.data == 0).nonzero().view(-1) 
            idx = (best_loss1.data[candidate]>loss1.data[candidate]).nonzero().view(-1)
            best_loss1[idx] = loss1[idx]


        idx = (best_loss1.data ==1000).nonzero().view(-1)
        if idx is None:
            return None
        else:
            return idx


    def __call__(self, input_xi, label_or_target, TARGETED=False):
        dis_a = []
        c_hi = 1000*torch.ones(input_xi.size()[0],1).cuda()
        c_lo = 0.01*torch.ones(c_hi.size()).cuda()
        while torch.max(c_hi-c_lo) > 1e-1:
            c_mid = (c_hi + c_lo)/2.0
            c_v = Variable(c_mid)
            idx = self.cw(input_xi, label_or_target, c_v, TARGETED)
            if idx is not None:
                c_hi[idx] = c_mid[idx]
                #print(c_mid)
            else:
                #dis = torch.norm(adv).data[0]
                #dis_a.append(dis)
                #print(dis)
                c_lo = c_mid
        return adv   
    
        
