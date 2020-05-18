import torch

class NES(object):
    def __init__(self,model):
        self.model=model

    def nes_grad_est(self,x, y, net, sigma=1e-3, n = 10):
        g = torch.zeros(x.size()).cuda()
        g = g.view(x.size()[0],-1)
        y = y.view(-1,1)
        for _ in range(n):
            u = torch.randn(x.size()).cuda()
            out1 = net.predict(x+sigma*u)
            out2 = net.predict(x-sigma*u)
            out1 = torch.gather(out1,1,y)
            #pdb.set_trace()
            out2 = torch.gather(out2,1,y)
            #print(out1.size(),u.size(),u.view(x.size()[0],-1).size())
            #print(out1[0][y],out2[0][y])
            g +=  out1 * u.view(x.size()[0],-1)
            g -=  out2 * u.view(x.size()[0],-1)
        g=g.view(x.size())
        return -1/(2*sigma*n) * g


    def nes(self, x_in, y, net, steps, eps, TARGETED):
        if eps == 0:
            return x_in
        x_adv = x_in.clone()
        lr = 0.01
        for i in range(steps):
            #print(f'\trunning step {i+1}/{steps} ...')
            # print(net.predict(x_adv)[0][y].item())
            if TARGETED:
                step_adv = x_adv - lr * torch.sign(self.nes_grad_est(x_adv, y, net))
            else:
                step_adv = x_adv + lr * torch.sign(self.nes_grad_est(x_adv, y, net))
            diff = step_adv - x_in
            diff.clamp_(-eps, eps)
            x_adv = x_in + diff
            x_adv.clamp_(0.0, 1.0)

        
        return x_adv

    def __call__(self, input_xi, label_or_target, TARGETED=False, epsilon=None):
        input_xi, label_or_target = input_xi.cuda(), label_or_target.cuda()
        return self.nes(input_xi,label_or_target,self.model, steps=100, eps=epsilon, TARGETED=TARGETED)