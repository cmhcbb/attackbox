from utils import mulvt
import time
import numpy as np 
from numpy import linalg as LA
import torch
import scipy
#from scipy import spacial
import scipy.spatial


learning_rate = 0.05
prev_contribution = 0
stopping = 0.001

class OPT_attack_lsq(object):
    def __init__(self,model):
        self.model = model
        self.prevX = None
        self.prevF = None

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
        least_square_query_count = query_count
        for i in range(iterations):
            gradient1, queries = self.eval_grad_least_squares(model, x0, y0, xg, initial_lbd = gg, 
                                                              tol=beta/500, h=0.01 ,percent_queries=0.1)
            least_square_query_count += queries
            if (i%3==0):
                gradient = self.eval_grad(model, x0, y0, xg, initial_lbd = gg, tol=beta/500, h=0.01)
                print("    Numerical - Approx gradient cosine distance: ", 
                      scipy.spatial.distance.cosine(gradient.flatten(), gradient1.flatten()))
            
            xg = xg - learning_rate * gradient1
            # print("Ditance moved: ", learning_rate * np.linalg.norm(gradient1))
            xg /= LA.norm(xg)
            gg, queries = self.fine_grained_binary_search_local(model, x0, y0, xg, initial_lbd = gg, tol=beta/500)
            least_square_query_count += queries
            print("Iteration: ", i, " Distortion: ", gg, " Queries: ", least_square_query_count)
            
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
              " %d queries %d \nTime: %.4f seconds" % (gg, target, least_square_query_count, timeend-timestart))
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

    def eval_grad_least_squares(self, model, x0, y0, theta, initial_lbd, tol=1e-5,  h=0.001, percent_queries=0.5):
        fx = initial_lbd # evaluate function value at original point
        x = theta
        D = x.flatten().shape[0]
        num_dirs = int(D*percent_queries)
        currF = np.zeros(num_dirs)
        currX = np.zeros([num_dirs, D])
        
        # if self.prevF is None or self.prevX is None:
        #     num_dirs *= 2
        A = np.zeros([num_dirs, D])
        B = np.zeros(num_dirs)
        queries = 0
        for i in range(num_dirs):
            u = np.random.randn(*x.shape)
            u /= np.linalg.norm(u)
            temp_x = x + h*u
            f1x, count = self.fine_grained_binary_search_local(model, x0, y0, temp_x, initial_lbd = initial_lbd, tol=h/500)
            # f1x, count = self.fine_grained_binary_search_local(model, x0, y0, theta, initial_lbd = initial_lbd, tol=h/500)
            df = (f1x - fx)/h
            A[i,:] = u.flatten()
            B[i] = df
            queries += count
            currF[i] = f1x
            currX[i] = temp_x.flatten()

        if prev_contribution > 0 and self.prevF is not None and self.prevX is not None:
            prevB = prev_contribution * (self.prevF - fx)
            prevA = prev_contribution * (self.prevX - x.flatten())
            A = np.concatenate((A, prevA), axis=0)
            B = np.concatenate((B, prevB))
            
        approx_grad = np.linalg.lstsq(A, B, rcond=None)[0]
        approx_grad = approx_grad.reshape(*x.shape)
        self.prevX = currX
        self.prevF = currF
        return approx_grad, queries 
    
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

    def __call__(self, input_xi, label_or_target, TARGETED=False, distortion=None):
        if TARGETED:
            print("Not Implemented.")
        else:
            adv = self.attack_untargeted(input_xi, label_or_target, distortion=distortion)
        return adv   
    
        
