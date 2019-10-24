import torch
import numpy as np
class NATTACK(object):
    def __init__(self, model, lr=0.01, delta=0.001, npop=300):
        self.model=model
        self.lr = lr
        self.delta = delta
        self.npop = npop

    def predict(self,x_in):
        x_t = torch.tensor(x_in).float().cuda()
        prob = self.model.predict_prob(x_t)
        return prob.cpu().numpy()
    
    def mapping_to_0_1(self,x):
        mean = (self.model.bounds[0]+self.model.bounds[1]) / 2.0
        var = (self.model.bounds[1]-self.model.bounds[0]) / 2.0
        return x*var + mean

    def mapping_to_1_1(self,x):
        mean = (self.model.bounds[0]+self.model.bounds[1]) / 2.0
        var = (self.model.bounds[1]-self.model.bounds[0]) / 2.0
        return (x-mean) / var

    def arctanh(self,x,eps=1e-6):
        x *= (1- eps)
        return (np.log((1+x)/(1-x)))*0.5

    def nattack(self, x_in, y, steps, eps, TARGETED):
        x_in = x_in.cpu().numpy()
        y = y.item()
        npop = 300
        sigma = 0.1
        alpha = 0.02
        modify = np.random.randn(*x_in.shape)
        for i in range(steps):
            #print(f'\trunning step {i+1}/{steps} ...')
            #print(net.predict(x_adv)[0][y].item())
            img_shape = x_in.shape[1:]
            #print(img_shape)
            Nsample = np.random.randn(npop, *img_shape)
            modifier = modify.repeat(npop,0) + sigma*Nsample
            x_r = self.arctanh(self.mapping_to_1_1(x_in)) # mapping to R 
            #x_1_1 = self.arctanh(np.tanh(x_in))  #?????
            x_adv = self.mapping_to_0_1(np.tanh(x_r+modifier))

            dist = np.clip(x_adv-x_in, -eps, eps)
            x_adv = (dist + x_in).reshape(npop,3,32,32)
            target_onehot = np.zeros((1,10))

            target_onehot[0][y]=1
            outputs = self.predict(x_adv)
            #print(clipinput.shape, outputs.shape)
            target_onehot = target_onehot.repeat(npop, 0)
            #print(np.argmax(outputs,axis=1),outputs.shape)
            if TARGETED:
                if (not np.any(np.argmax(outputs,axis=1)!= y)) and (np.abs(dist).max() <= eps):
                    return x_adv, np.argwhere(np.argmax(outputs,axis=1)== y) 
            else:
                if (np.any(np.argmax(outputs,axis=1)!= y)) and (np.abs(dist).max() <= eps):
                    #print(np.argmax(outputs,axis=1))
                    return x_adv, np.argwhere(np.argmax(outputs,axis=1)!= y) 
            real = np.log((target_onehot * outputs).sum(1)+1e-30)
            other = np.log(((1 - target_onehot)*outputs - target_onehot * 10000).max(1)[0]+1e-30)
            loss1 = np.clip(real-other,0, 1000)
            reward = -0.5 * loss1
            A = (reward - np.mean(reward)) / (np.std(reward)+1e-7)

            if TARGETED:
                modify -= (alpha/(npop*sigma)) * ((np.dot(Nsample.reshape(npop,-1).T,A)).reshape(3,32,32))
            else:
                modify += (alpha/(npop*sigma)) * ((np.dot(Nsample.reshape(npop,-1).T,A)).reshape(3,32,32))
        return x_adv, None

    def __call__(self, input_xi, label_or_target, TARGETED=False):
        input_xi, label_or_target = input_xi.cuda(), label_or_target.cuda()
        x_adv,index = self.nattack(input_xi,label_or_target,steps=500, eps=0.031, TARGETED=TARGETED)
        if index is None:
            return input_xi
        index = np.squeeze(index)
        #print(index)
        return torch.tensor(x_adv[index]).float().cuda()
