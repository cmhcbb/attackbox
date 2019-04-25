from utils import mulvt
import time
import numpy as np 
from numpy import linalg as LA
import torch
import scipy.spatial

start_learning_rate = 1.0
stopping = 0.001

def sign(y):
    """
    y -- numpy array of shape (m,)
    Returns an element-wise indication of the sign of a number.
    The sign function returns -1 if y < 0, 1 if x >= 0. nan is returned for nan inputs.
    """
    y_sign = np.sign(y)
    y_sign[y_sign==0] = 1
    return y_sign

class OPT_attack_sign_SGD(object):
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

        # Calculate a good starting point.
        print("Running gradient descent with start learning rate ", start_learning_rate)
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
    
    
        # Begin Gradient Descent.
        timestart = time.time()
        xg, gg = best_theta, g_theta
        learning_rate = start_learning_rate
        prev_obj = 100000
        distortions = [gg]
        for i in range(iterations):
            sign_gradient, grad_queries = self.sign_grad_v1(x0, y0, xg, initial_lbd=gg)
            
            if False:
                print("simran")
                # Compare cosine distance with numerical gradient.
                gradient, _ = self.eval_grad(model, x0, y0, xg, initial_lbd=gg, tol=beta/500, h=0.01)
                print("    Numerical - Sign gradient cosine distance: ", 
                      scipy.spatial.distance.cosine(gradient.flatten(), sign_gradient.flatten()))
            
            # Learning rate decay
            if i!=0 and i%20==0:
                learning_rate *= 0.9
                
            xg = xg - learning_rate * sign_gradient
            xg /= LA.norm(xg)
            
            gg, gg_queries = self.fine_grained_binary_search_local(model, x0, y0, xg, initial_lbd = gg, tol=beta/500)
            query_count += (grad_queries + gg_queries)
            distortions.append(gg)
            
            if i%5==0:
                print("Iteration: ", i, " Distortion: ", gg, " Queries: ", query_count, " LR: ", learning_rate)
            
            if distortion is not None and gg < distortion:
                print("Success: required distortion reached")
                break

            if gg > prev_obj-stopping:
                print("Success: stopping threshold reached")
                #learning_rate *= 0.7
                break
            
            prev_obj = gg

        target = model.predict_label(x0 + torch.tensor(gg*xg, dtype=torch.float).cuda())
        timeend = time.time()
        print("\nAdversarial Example Found Successfully: distortion %.4f target"
              " %d queries %d \nTime: %.4f seconds" % (gg, target, query_count, timeend-timestart))
        print("Distortions: ", distortions)
        return x0 + torch.tensor(gg*xg, dtype=torch.float).cuda(), gg

    def sign_grad_v1(self, x0, y0, theta, initial_lbd, h=0.001, K=500, lr=5.0):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        sign_grad = np.zeros(theta.shape)
        queries = 0
        for _ in range(K):
            u = np.random.randn(*theta.shape)
            u /= LA.norm(u)
            
            sign = -1
            new_theta = theta + h*u
            new_theta /= LA.norm(new_theta)
            if self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).cuda()) == y0:
                sign = 1
            queries += 1
            sign_grad += u*sign
        sign_grad /= K
        
        # sign_grad /= LA.norm(sign_grad)
        # new_theta = theta + h*sign_grad
        # new_theta /= LA.norm(new_theta)
        # fxph, q1 = self.fine_grained_binary_search_local(self.model, x0, y0, new_theta, initial_lbd=initial_lbd, tol=h/500)
        # delta = (fxph - initial_lbd)/h
        # queries += q1
        # sign_grad *= lr*delta
        
        
        
        return sign_grad, queries
    
    def sign_grad_v2(self, x0, y0, theta, initial_lbd, h=0.001, K=20):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        sign_grad = np.zeros(theta.shape)
        queries = 0
        for _ in range(K):
            u = np.random.randn(*theta.shape)
            u /= LA.norm(u)
            
            sign = -1
            new_theta = theta + h*u
            new_theta /= LA.norm(new_theta)
            if self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).cuda()) == y0:
                sign = 1
            queries += 1
            sign_grad += sign(u)*sign
        sign_grad /= K
        return sign_grad, queries

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

    def eval_grad(self, model, x0, y0, theta, initial_lbd, tol=1e-5,  h=0.001, sign=False):
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
            unit_x = x / LA.norm(x)
            if sign:
                if model.predict_label(x0+torch.tensor(initial_lbd*unit_x, dtype=torch.float).cuda()) == y0:
                    g = 1
                else:
                    g = -1
                q1 = 1
            else:
                fxph, q1 = self.fine_grained_binary_search_local(model, x0, y0, unit_x, initial_lbd = initial_lbd, tol=h/500)
                g = (fxph - fx) / (h)
            
            queries += q1
            # x[ix] = oldval - h
            # fxmh, q2 = self.fine_grained_binary_search_local(model, x0, y0, x, initial_lbd = initial_lbd, tol=h/500)
            x[ix] = oldval # restore

            # compute the partial derivative with centered formula
            grad[ix] = g
            it.iternext() # step to next dimension

        # print("Found gradient")
        return grad, queries

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
    
        
