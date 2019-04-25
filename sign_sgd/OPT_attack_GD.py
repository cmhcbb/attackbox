from utils import mulvt
import time
import numpy as np 
from numpy import linalg as LA
import torch

learning_rate = 0.05
stopping = 0.001

class OPT_attack_GD(object):
    def __init__(self,model):
        self.model = model

    def attack_untargeted(self, x0, y0, alpha = 0.2, beta = 0.001, iterations = 1000, distortion=None):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """
        model = self.model
        y0 = y0[0]
        query_count = 0
        
        if (model.predict_label(x0) != y0):
            print("Fail to classify the image. No need to attack.")
            return x0

        print("Running gradient descent with learning rate ", learning_rate)
        num_directions = 100
        best_theta, g_theta = None, float('inf')
        print("Searching for the initial direction on %d random directions: " % (num_directions))
        timestart = time.time()
        for i in range(num_directions):
            query_count += 1
            theta = np.random.randn(*x0.shape)
            if model.predict_label(x0+torch.tensor(theta, dtype=torch.float).cuda())!=y0:
                initial_lbd = LA.norm(theta)
                theta /= initial_lbd
                lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    print("--------> Found distortion %.4f" % g_theta)

        timeend = time.time()
        print("==========> Found best distortion %.4f in %.4f seconds "
              "using %d queries" % (g_theta, timeend-timestart, query_count))
    
        timestart = time.time()
        xg, gg = best_theta, g_theta
        prev_obj = 100000
        for i in range(iterations):
            gradient, grad_queries = self.eval_grad(model, x0, y0, xg, initial_lbd = gg, tol=beta/500)
            print("Gradient norm: ", LA.norm(gradient))
            xg = xg - learning_rate * gradient
            xg /= LA.norm(xg)
            
            gg, gg_queries = self.fine_grained_binary_search_local(model, x0, y0, xg, initial_lbd = gg, tol=beta/500)
            query_count += (grad_queries + gg_queries)
            print("Iteration: ", i, " Distortion: ", gg, " Queries: ", query_count)
            
            if distortion is not None:
                if gg < distortion:
                    print("Success: required distortion reached")
                    break

            if gg > prev_obj-stopping:
                print("Success: stopping threshold reached")
                break
            
            prev_obj = gg

        target = model.predict_label(x0 + torch.tensor(gg*xg, dtype=torch.float).cuda())
        timeend = time.time()
        print("\nAdversarial Example Found Successfully: distortion %.4f target"
              " %d queries %d \nTime: %.4f seconds" % (gg, target, query_count, timeend-timestart))
        return x0 + torch.tensor(gg*xg, dtype=torch.float).cuda(), gg

    def fine_grained_binary_search_local(self, model, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
         
        if model.predict_label(x0+torch.tensor(lbd*theta, dtype=torch.float).cuda()) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while model.predict_label(x0+torch.tensor(lbd_hi*theta, dtype=torch.float).cuda()) == y0:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while model.predict_label(x0+torch.tensor(lbd_lo*theta, dtype=torch.float).cuda()) != y0 :
                lbd_lo = lbd_lo*0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + torch.tensor(lbd_mid*theta, dtype=torch.float).cuda()) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def fine_grained_binary_search(self, model, x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best: 
            if model.predict_label(x0+torch.tensor(current_best*theta, dtype=torch.float).cuda()) == y0:
                nquery += 1
                return float('inf'), nquery
            lbd = current_best
        else:
            lbd = initial_lbd
        
        lbd_hi = lbd
        lbd_lo = 0.0

        while (lbd_hi - lbd_lo) > 1e-5:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + torch.tensor(lbd_mid*theta, dtype=torch.float).cuda()) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def eval_grad(self, model, x0, y0, theta, initial_lbd, tol=1e-5,  h=0.001):
        # print("Finding gradient")
        fx = initial_lbd # evaluate function value at original point
        grad = np.zeros_like(theta)
        x = theta
        # iterate over all indexes in x
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        
        queries = 0
        
        while not it.finished:

            # evaluate function at x+h
            ix = it.multi_index
            oldval = x[ix]
            x[ix] = oldval + h # increment by h
            fxph, q1 = self.fine_grained_binary_search_local(model, x0, y0, x, initial_lbd = initial_lbd, tol=h/500)
            queries += q1
            # x[ix] = oldval - h
            # fxmh, q2 = self.fine_grained_binary_search_local(model, x0, y0, x, initial_lbd = initial_lbd, tol=h/500)
            x[ix] = oldval # restore

            # compute the partial derivative with centered formula
            grad[ix] = (fxph - fx) / (h) # the slope
            it.iternext() # step to next dimension

        # print("Found gradient")
        return grad, queries

    def __call__(self, input_xi, label_or_target, TARGETED=False, distortion=None):
        if TARGETED:
            print("Not Implemented.")
        else:
            adv = self.attack_untargeted(input_xi, label_or_target, distortion=distortion)
        return adv   
    
        
