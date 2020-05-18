import time
import numpy as np 
from numpy import linalg as LA
from models import PytorchModel
import torch, random
from allmodels import MNIST, load_model, load_mnist_data, load_cifar10_data, CIFAR10

class OPT_genattack(object):
    def __init__(self,model):
        self.model = model
    
    def genattack_untargeted(self, x0, y0, alpha = 0.2, population = 10):
        model = self.model
        print(x0.shape)
        y0 = y0[0]
        if (model.predict_label(x0) != y0):
            print("Fail to classify the image. No need to attack.")
            return x0
        #d_shape = d_shape[np.newaxis, ...]
        d_shape = (population,1,28,28)
        directions = np.random.randn(*d_shape)
        next_directions = np.random.randn(*d_shape)
        initial_lbd = np.zeros(population)
        fit = np.zeros(population)
        max_g = 1000
        def crossover(parent1,parent2,initial_p1,initial_p2):
            parent1, parent2 = parent1/initial_p1, parent2/initial_p2
            fit_p1, count_p1 = self.fine_grained_binary_search(model, x0, y0, parent1, initial_p1)
            fit_p2, count_p2 = self.fine_grained_binary_search(model, x0, y0, parent2, initial_p2)
            p1 = fit_p1/(fit_p1+fit_p2)
            mask = np.random.binomial(1,p1,28*28)
            child_f = parent1.flatten()*mask + parent2.flatten()*(1-mask)
            return child_f.reshape(1,1,28,28), count_p1+count_p2
        sum_count = 0
        for iter in range(max_g):
            for i in range(population):
                theta  = directions[i].copy().reshape(1,1,28,28)
                initial_lbd[i] = LA.norm(theta)
                theta /= initial_lbd[i]
                fit[i], count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd[i])
                sum_count += count
                #directions[i] = theta.reshape(1,28,28)
            
            best_idx = np.argmin(fit)
            best_fit = fit[best_idx]
            best_theta = directions[best_idx] 
            if best_fit<2:
                return best_theta
            next_directions[0] = best_theta
            #probs = fit/LA.norm(fit)
            sum_fit = np.sum(fit)
            probs = fit/sum_fit
            #print(probs)
            for i in range(1,population):
                parent_idx= np.random.choice(10,2,p=probs)
                parent1, initial_p1 = directions[parent_idx[0]], initial_lbd[parent_idx[0]]
                parent2, initial_p2 = directions[parent_idx[1]], initial_lbd[parent_idx[1]]
                child, count1 = crossover(parent1,parent2, initial_p1,initial_p2)
                sum_count += count1
                mask = np.random.binomial(1,0.05,28*28)
                child_mut = child + (np.random.randn(1,1,28,28).flatten()*mask).reshape(1,1,28,28)
                next_directions[i] = child_mut
            #print("Iter==================")
            #print(LA.norm(next_directions[1]))
            #print(LA.norm(directions[1]))
            directions = next_directions.copy()
            if iter%1==0:
                print(best_fit, sum_count)
        return x0+best_theta*fit[0]

    def fine_grained_binary_search_local(self, model, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
         
        if model.predict_label(x0+lbd*theta) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while model.predict_label(x0+lbd_hi*theta) == y0:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while model.predict_label(x0+lbd_lo*theta) != y0 :
                lbd_lo = lbd_lo*0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + lbd_mid*theta) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def fine_grained_binary_search(self, model, x0, y0, theta, initial_lbd):
        nquery = 0
        #if initial_lbd > current_best: 
        #    if model.predict_label(x0+current_best*theta) == y0:
        #        nquery += 1
        #       return float('inf'), nquery
        #    lbd = current_best
        #else:
        #    lbd = initial_lbd
        
        lbd_hi = initial_lbd
        lbd_lo = 0.0

        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + lbd_mid*theta) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery


    def __call__(self, input_xi, label_or_target, TARGETED=False):
        if TARGETED:
            print("Not Implemented.")
        else:
            adv = self.genattack_untargeted(input_xi, label_or_target)
        return adv   
    
        
if __name__ == '__main__':
    #timestart = time.time()
    random.seed(0)
    net = MNIST()
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0])
    load_model(net,'mnist_gpu.pt')
    net.eval()
    model = net.module if torch.cuda.is_available() else net
    amodel = PytorchModel(model, bounds=[0,1], num_classes=10)
    attack = OPT_genattack(amodel)
    train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()
    for i, (xi,yi) in enumerate(test_loader):
        if i==1:
            break
    xi = xi.numpy()
    attack(xi,yi)