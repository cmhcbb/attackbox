from utils import mulvt
import time
import numpy as np 
from numpy import linalg as LA
import torch
import scipy

learning_rate = 0.01

class OPT_attack_polar_GD(object):
    def __init__(self,model):
        self.model = model

    def attack_untargeted(self, x0, y0, alpha = 0.2, beta = 0.001, iterations = 1000):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """
        model = self.model
        y0 = y0[0]
        if (model.predict_label(x0) != y0):
            print("Fail to classify the image. No need to attack.")
            return x0

        print("Running gradient descent with learning rate ", learning_rate)
        num_directions = 100
        best_theta, g_theta = None, float('inf')
        query_count = 0
        grad_queries = 0
        # D = *x0.flatten().shape
        # polar_shape = D-1
        # theta_shape = *x0.shape
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
        g1 = 1.0
        theta, g2 = best_theta, g_theta
        _, theta_polar = self.ct_to_polar(theta)
        best_theta_polar = theta_polar
        opt_count = 0
        stopping = 0.01
        prev_obj = 100000
        
        _, xg = self.ct_to_polar(theta)
        gg = np.copy(g2)
        
        for i in range(iterations):
            gradient, grad_queries = self.eval_grad(model, x0, y0, xg, initial_lbd = gg, tol=beta/500)
            xg = xg - learning_rate * gradient
            temp_theta = self.polar_to_ct(xg).reshape(*x0.shape)
            gg, gg_queries = self.fine_grained_binary_search_local(model, x0, y0, temp_theta, initial_lbd = gg, tol=beta/500)
            query_count += (gg_queries + grad_queries)
            xg = self.correct_polar(xg)
            print("Iteration: ", i, " Distortion: ", gg, " Queries: ", query_count)

            # Modify convergence criteria
            # if (i+1)%10 == 0:
            #     print("Iteration %3d: g(theta + beta*u) = %.8f g(theta) = %.4f "
            #           "distortion %.8f num_queries %d" % (i+1, g1, g2, LA.norm(g2*theta), opt_count))
            #     # print("Simmy")
            #     if g2 > prev_obj-stopping:
            #         print("success")
            #         break
            #     prev_obj = g2
            continue
            

        temp_theta = self.polar_to_ct(best_theta_polar).reshape(*x0.shape)
        target = model.predict_label(x0 + torch.tensor(g_theta*temp_theta, dtype=torch.float).cuda())
        timeend = time.time()
        print("\nAdversarial Example Found Successfully: distortion %.4f target"
              " %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
        return x0 + torch.tensor(g_theta*temp_theta, dtype=torch.float).cuda()

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

    def eval_grad(self, model, x0, y0, theta_polar, initial_lbd, tol=1e-5,  h=0.001):
        fx = initial_lbd # evaluate function value at original point
        grad = np.zeros_like(theta_polar)
        x = theta_polar
        # iterate over all indexes in x
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        queries = 0
        while not it.finished:

            # evaluate function at x+h
            ix = it.multi_index
            oldval = x[ix]
            x[ix] = oldval + h # increment by h
            theta_ct = self.polar_to_ct(x).reshape(*x0.shape)
            fxph, q1 = self.fine_grained_binary_search_local(model, x0, y0, theta_ct, initial_lbd = initial_lbd, tol=h/500)
            queries += q1
            # x[ix] = oldval - h
            # theta = self.polar_to_ct(x).reshape(*x0.shape)
            # fxmh, _ = self.fine_grained_binary_search_local(model, x0, y0, theta, initial_lbd = initial_lbd, tol=h/500)
            x[ix] = oldval # restore

            # compute the partial derivative with centered formula
            # grad[ix] = (fxph - fxmh) / (2 * h) # the slope
            grad[ix] = (fxph - fx) / (h) # the slope
            # if True:
            #     print(ix, grad[ix])
            it.iternext() # step to next dimension

        return grad, queries

    def correct_polar(self, theta):
        """
        Correct polar angle if out of domain.
        """
        new_theta_polar = theta
        neg1 = theta[theta<0]
        theta1 = theta[:-1]
        neg2 = theta1[theta1>np.pi]
        if len(neg1)>0 or len(neg2)>0:
            temp_theta = self.polar_to_ct(theta)
            _, new_theta_polar = self.ct_to_polar(temp_theta)
        return new_theta_polar
            
    def polar_to_ct(self, arr, r=1):
        """
        Convert polar to cartesian coordinates.
        """
        a = np.concatenate((np.array([2*np.pi]), arr))
        si = np.sin(a)
        si[0] = 1
        si = np.cumprod(si)
        co = np.cos(a)
        co = np.roll(co, -1)
        return si*co*r

    def ct_to_polar(self, arr, eps=1e-5):
        arr1 = np.copy(np.array(arr))
        arr1 = arr1.flatten()
        arr2 = arr1**2 + eps
        r = np.sqrt(np.sum(arr2))
        arr2 = np.flip(np.cumsum(np.flip(arr2, axis=0)), axis=0)
        arr2 = np.sqrt(arr2)
        phi = np.arccos((arr1/arr2)[:-1])
        if arr1[-1] < 0:
            phi[-1] = 2*np.pi - phi[-1]
        # phi = np.concatenate((np.array([r]), phi))
        return r, phi
    
    def __call__(self, input_xi, label_or_target, TARGETED=False):
        if TARGETED:
            print("Not Implemented.")
        else:
            adv = self.attack_untargeted(input_xi, label_or_target)
        return adv   
    
        
