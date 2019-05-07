from utils import mulvt
import time
import numpy as np 
from numpy import linalg as LA
import torch

class OPT_attack(object):
    def __init__(self,model):
        self.model = model

    def attack_untargeted(self, x0, y0, alpha = 0.2, beta = 0.001, iterations = 1000, distortion=None, seed=None):
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

        if seed is not None:
            np.random.seed(seed)
        num_directions = 100
        best_theta, g_theta = None, float('inf')
        query_count = 0
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
        opt_count = 0
        stopping = 0.005
        prev_obj = 100000
        v_gradient = np.zeros(theta.shape)
        xg = np.copy(theta)
        gg = np.copy(g2)
        for i in range(iterations):            
            gradient = np.zeros(theta.shape)
            q = 10
            min_g1 = float('inf')
            for _ in range(q):
                u = np.random.randn(*theta.shape)
                u /= LA.norm(u)
                
                # if LA.norm(v_gradient) < 1e-8:
                #     mu = v_gradient.flatten()
                # else:
                #     mu = v_gradient.flatten()/LA.norm(v_gradient)
                # u = np.random.multivariate_normal(mu, np.identity(mu.shape[0]))
                # u = u.reshape(theta.shape)/LA.norm(u)
                # if LA.norm(v_gradient) > 1e-8:
                #     u += v_gradient/LA.norm(v_gradient)
                # u /= LA.norm(u)
                
                ttt = theta+beta * u
                ttt /= LA.norm(ttt)
                g1, count = self.fine_grained_binary_search_local(model, x0, y0, ttt, initial_lbd = g2, tol=beta/500)
                opt_count += count
                gradient += (g1-g2)/beta * u
                if g1 < min_g1:
                    min_g1 = g1
                    min_ttt = ttt
            gradient = 1.0/q * gradient
#             v_gradient = gradient/LA.norm(gradient)
            
#             new_dir = gradient/LA.norm(gradient)
#             v_gradient = 0.9 * v_gradient + 0.1 * gradient
#             v_gradient /= LA.norm(v_gradient)

            if distortion is not None:
                if g2 < distortion:
                    print("Success: required distortion reached")
                    break

            if (i+1)%50 == 0:
                print("Iteration %3d: g(theta + beta*u) = %.8f g(theta) = %.4f "
                      "distortion %.8f num_queries %d" % (i+1, g1, g2, LA.norm(g2*theta), opt_count))
                # print("Simmy")
                if g2 > prev_obj-stopping:
                    print("Success: Success: stopping threshold reached")
                    break
                prev_obj = g2

            min_theta = theta
            min_g2 = g2
            # print("original alpha ", alpha)
            for _ in range(15):
                new_theta = theta - alpha * gradient
                new_theta /= LA.norm(new_theta)
                new_g2, count = self.fine_grained_binary_search_local(
                    model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                opt_count += count
                alpha = alpha * 2
                # print("alpha1 ", alpha)
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                else:
                    break
            # print("alpha after increment ", alpha)
            if min_g2 >= g2:
                for _ in range(15):
                    alpha = alpha * 0.25
                    # print("alpha2 ", alpha)
                    new_theta = theta - alpha * gradient
                    new_theta /= LA.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local(
                        model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                    opt_count += count
                    if new_g2 < g2:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        break
                        
            # print("alpha after decrement ", alpha)
            if min_g2 <= min_g1:
                theta, g2 = min_theta, min_g2
            else:
                theta, g2 = min_ttt, min_g1

            if g2 < g_theta:
                best_theta, g_theta = theta, g2
            
            #print(alpha)
            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
                beta = beta * 0.1
                if (beta < 0.0005):
                    break

        target = model.predict_label(x0 + torch.tensor(g_theta*best_theta, dtype=torch.float).cuda())
        timeend = time.time()
        print("\nAdversarial Example Found Successfully: distortion %.4f target"
              " %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
        return x0 + torch.tensor(g_theta*best_theta, dtype=torch.float).cuda(), g_theta

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
        while not it.finished:

            # evaluate function at x+h
            ix = it.multi_index
            oldval = x[ix]
            x[ix] = oldval + h # increment by h
            fxph, _ = self.fine_grained_binary_search_local(model, x0, y0, x, initial_lbd = initial_lbd, tol=h/500)
            x[ix] = oldval - h
            fxmh, _ = self.fine_grained_binary_search_local(model, x0, y0, x, initial_lbd = initial_lbd, tol=h/500)
            x[ix] = oldval # restore

            # compute the partial derivative with centered formula
            grad[ix] = (fxph - fxmh) / (2 * h) # the slope
            # if True:
            #     print(ix, grad[ix])
            it.iternext() # step to next dimension

        # print("Found gradient")
        return grad

    def polar_to_ct(arr, r=1):
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

    def __call__(self, input_xi, label_or_target, TARGETED=False, distortion=None, seed=None):
        if TARGETED:
            print("Not Implemented.")
        else:
            adv = self.attack_untargeted(input_xi, label_or_target, distortion=distortion, seed=seed)
        return adv   
    
        
