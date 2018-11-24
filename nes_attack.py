import torch
from torch.autograd import Variable
import torch.optim as optim
from utils import mulvt
zero_iters=50
label_only_sigma = 0.5
batch_per_gpu = 4
sigma = 0.001
momentum = 0.1
min_lr = 5e-5
goal_epsilon = 0.1
delta_e = 0.01
class NES(object):
    def __init__(self,model):
        self.model = model

    def one_hot(yi):
        label_onehot = torch.FloatTensor(yi.size()[0],self.model.num_classes)
        label_onehot.zero_()
        label_onehot.scatter_(1,yi.view(-1,1),1)
        return label_onehot        

    def get_loss(self,eval_points, input_size, target):
        tiled_points = eval_points.unsqueeze(0).expand((zero_iters,)+eval_points.size())
        random_noise = torch.randn(tiled_points.size()).cuda()
        random_noise /= torch.norm(random_noise)
        noised_eval_im = tiled_points + random_noise * label_only_sigma
        output = self.model.predict_label_batch(noised_eval_im.view((-1,)+input_size))
        batches_in = (output == target).view(zero_iters,batch_per_gpu,1)
        return 1 - torch.mean(torch.mean(batches_in.float(),2),0)

    def get_gradient(self,input_xi,target):
        input_size = input_xi.size()
        noise_pos = torch.randn((batch_per_gpu//2,)+input_size).cuda()
        noise = torch.cat((noise_pos,-noise_pos),0)
        #print(noise.size())
        eval_points = input_xi + noise* sigma 
        loss = self.get_loss(eval_points,input_size, target)
        #print(loss)
        #print(noise.size())
        loss_tiled = loss.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(noise.size())
        #print(loss_tiled.size())
        gradient = torch.mean(loss_tiled*noise,0)/sigma
        return gradient
    
    def nes(self, input_xi, initial_img, target, TARGETED=False):    
        #label_onehot = one_hot(target)
        delta_e = 0.01
        xi = torch.from_numpy(input_xi).cuda()
        adv = torch.from_numpy(initial_img).cuda()
        #print(xi.size())
        g = 0
        initial_epsilon = torch.norm(xi-adv)/2
        epsilon = 0.5
        adv = xi + epsilon * (adv-xi)
        for it in range(1000):
            #epsilon = initial_epsilon
            print(epsilon)
            if self.model.predict_label(adv)==target and epsilon< goal_epsilon:
                print("early stop")
                break

            prev_g = g
            g = self.get_gradient(xi,target)
            g = momentum * prev_g + (1.0 - momentum) * g

            current_lr = 0.01
            proposed_adv = adv - current_lr * g
            while self.model.predict_label(proposed_adv)!=target:
                if current_lr<min_lr:
                    epsilon += delta_e
                    delta_e /= 2
                    proposed_adv = adv
                    #print("couldn't find a valid lr, increase the epsilon") 
                    break 

                current_lr /= 2 
                proposed_adv = adv - current_lr * g
                proposed_adv = xi + epsilon * (proposed_adv-xi)

            adv = proposed_adv
            epsilon -= delta_e
        print(self.model.predict_label(proposed_adv), torch.norm(adv-xi))
        return adv
    def __call__(self, input_xi, initial_img, target, TARGETED=True):
        adv = self.nes(input_xi, initial_img, target , TARGETED)
        return adv
    
        
