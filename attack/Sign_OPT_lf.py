from os import linesep
from pickle import NONE
from matplotlib import lines
from matplotlib.pyplot import errorbar, fignum_exists
import numpy as np
from numpy.core.numeric import indices
import torch
import scipy.spatial
import random, logging, time

#from qpsolvers import solve_qp
from numpy import linalg as LA
from scipy.linalg import qr
from torch import t

start_learning_rate = 1.0

def quad_solver(Q, b):
    """
    Solve min_a  0.5*aQa + b^T a s.t. a>=0
    """
    K = Q.shape[0]
    alpha = np.zeros((K,))
    g = b
    Qdiag = np.diag(Q)
    for i in range(20000):
        delta = np.maximum(alpha - g/Qdiag,0) - alpha
        idx = np.argmax(abs(delta))
        val = delta[idx]
        if abs(val) < 1e-7: 
            break
        g = g + val*Q[:,idx]
        alpha[idx] += val
    return alpha

def sign(y):
    """
    y -- numpy array of shape (m,)
    Returns an element-wise indication of the sign of a number.
    The sign function returns -1 if y < 0, 1 if x >= 0. nan is returned for nan inputs.
    """
    y_sign = np.sign(y)
    y_sign[y_sign==0] = 1
    return y_sign


class OPT_attack_sign_SGD_lf(object):
    def __init__(self, model, k=200, train_dataset=None):
        self.model = model
        self.k = k
        self.train_dataset = train_dataset

    def attack_untargeted(self, x0, y0, alpha = 0.2, beta = 0.001, iterations = 1000, query_limit=80000,
                          distortion=None, stopping=1e-8):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """
        model = self.model
        y0 = y0[0]
        query_count = 0
        ls_total = 0
        
        if (model.predict_label(x0) != y0):
            logging.info("Fail to classify the image. No need to attack.")
            return x0, 0, True, 0, None

        #### init theta by Gaussian: Calculate a good starting point.
        num_directions = 100
        best_theta, g_theta = None, float('inf')
        init_thetas, init_g_thetas = [], []

        for i in range(num_directions):
            query_count += 1
            theta = np.random.randn(*x0.shape)

            if model.predict_label(x0+torch.tensor(theta, dtype=torch.float).cuda()) != y0:
                initial_lbd = LA.norm(theta.flatten(), np.inf)
                theta /= initial_lbd

                lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    init_thetas.append(best_theta)
                    init_g_thetas.append(g_theta)

        logging.info("Best initial distortion: {:.3f}".format(g_theta))

        ## fail if cannot find a adv direction within 200 Gaussian
        if g_theta == float('inf'):
            logging.info("Couldn't find valid initial, failed")
            return x0, 0, False, query_count, None

        for g_theta, best_theta in zip(init_g_thetas, init_thetas): # successful attack upon initialization
            if distortion is None or g_theta < distortion:
                x_adv = x0 + torch.tensor(g_theta*best_theta, dtype=torch.float).cuda()
                target = model.predict_label(x_adv)
                logging.info("\nSucceed: distortion {:.4f} target"
                    " {:d} queries {:d} LS queries {:d}".format(g_theta, target, query_count, 0))
                return x0 + torch.tensor(x_adv, dtype=torch.float).cuda(), g_theta, True, query_count, best_theta


        #### begin attack
        init_thetas = list(reversed(init_thetas))
        init_g_thetas = list(reversed(init_g_thetas))
        query_count_init = query_count
        ls_total_init = ls_total
        alpha_init = alpha
        beta_init = beta
        for init_id in range(1):
            best_theta = init_thetas[init_id]
            g_theta = init_g_thetas[init_id]
            ls_total = ls_total_init
            query_count = query_count_init
            alpha = alpha_init
            beta  = beta_init

            #### Begin Gradient Descent.
            xg, gg = best_theta, g_theta
            distortions = [gg]
            for i in range(iterations):
                sign_gradient, grad_queries = self.sign_grad_v1(x0, y0, xg, initial_lbd=gg, h=beta)

                ## Line search of the step size of gradient descent)
                min_theta, min_g2, alpha, ls_count = self.line_search(model, x0, y0, gg, xg, alpha, sign_gradient, beta)

                if alpha < 1e-6:
                    alpha = 1.0
                    logging.info("Warning: not moving, beta is {0}".format(beta))
                    beta = beta * 0.1
                    if (beta < 1e-8):
                        break

                xg, g2 = min_theta, min_g2
                gg = g2

                query_count += (grad_queries + ls_count)
                ls_total += ls_count
                distortions.append(gg)

                if query_count > query_limit:
                    break
                
                if i % 5 == 0:
                    logging.info("Iteration {:3d} distortion {:.6f} num_queries {:d}".format(i+1, gg, query_count))

                if distortion is not None and gg < distortion:
                    logging.info("Success: required distortion reached")
                    break

            ## check if successful
            if distortion is None or gg < distortion:
                target = model.predict_label(x0 + torch.tensor(gg*xg, dtype=torch.float).cuda())
                logging.info('Succeed at init {}'.format(init_id))
                logging.info("Succeed distortion {:.4f} target"
                             " {:d} queries {:d} LS queries {:d}\n".format(gg, target, query_count, ls_total))
                return x0 + torch.tensor(gg*xg, dtype=torch.float).cuda(), gg, True, query_count, xg
            
        return x0, 0, False, query_count, xg


    def line_search(self, model, x0, y0, gg, xg, alpha, sign_gradient, beta):
        ls_count = 0
        min_theta = xg
        min_g2 = gg
        for _ in range(15):
            new_theta = xg - alpha * sign_gradient
            new_theta /= LA.norm(new_theta.flatten(), np.inf)
            new_g2, count = self.fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
            ls_count += count
            alpha = alpha * 2
            if new_g2 < min_g2:
                min_theta = new_theta
                min_g2 = new_g2
            else:
                break

        if min_g2 >= gg:
            for _ in range(15):
                alpha = alpha * 0.25
                new_theta = xg - alpha * sign_gradient
                new_theta /= LA.norm(new_theta.flatten(), np.inf)
                new_g2, count = self.fine_grained_binary_search_local(
                    model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                ls_count += count
                if new_g2 < gg:
                    min_theta = new_theta
                    min_g2 = new_g2
                    break
        return min_theta, min_g2, alpha, ls_count


    def sign_grad_v1(self, x0, y0, theta, initial_lbd, h=0.001, lr=5.0, D=4, target=None,
                     sample_type='gaussian'):
        """
        Evaluate the sign of gradient by formulat
        sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]
        """
        K = self.k
        sign_grad = np.zeros(theta.shape)
        queries = 0

        for iii in range(K):
            if sample_type == 'gaussian':
                u = np.random.randn(*theta.shape)
            else:
                logging.info('ERROR: UNSUPPORTED SAMPLE_TYPE: {}'.format(sample_type)); exit(1)
            u /= LA.norm(u.flatten(), np.inf)

            new_theta = theta + h*u;
            new_theta /= LA.norm(new_theta.flatten(), np.inf)
            sign = 1

            # Targeted case.
            if (target is not None and 
                self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).cuda()) == target):
                sign = -1

            # Untargeted case
            if (target is None and
                self.model.predict_label(x0+torch.tensor(initial_lbd*new_theta, dtype=torch.float).cuda()) != y0): # success
                sign = -1
            queries += 1
            sign_grad += u*sign
        sign_grad /= K
        
        return sign_grad, queries

     
    ##########################################################################################

    def fine_grained_binary_search_local(self, model, x0, y0, theta, initial_lbd = 1.0, lin_search_ratio=0.01, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
        tol = max(tol, 2e-9)
        if model.predict_label(x0+torch.tensor(lbd*theta, dtype=torch.float).cuda()) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*(1 + lin_search_ratio)
            nquery += 1
            while model.predict_label(x0+torch.tensor(lbd_hi*theta, dtype=torch.float).cuda()) == y0:
                lbd_hi = lbd_hi*(1 + lin_search_ratio)
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*(1 - lin_search_ratio)
            nquery += 1
            while model.predict_label(x0+torch.tensor(lbd_lo*theta, dtype=torch.float).cuda()) != y0 :
                lbd_lo = lbd_lo*(1 - lin_search_ratio)
                nquery += 1
                
        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + torch.tensor(lbd_mid*theta, dtype=torch.float).cuda()) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery


    def fine_grained_binary_search_local(self, model, x0, y0, theta, initial_lbd = 1.0, lin_search_ratio=0.01, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
        tol = max(tol, 2e-9)
        if model.predict_label(x0+torch.tensor(lbd*theta, dtype=torch.float).cuda()) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*(1 + lin_search_ratio)
            nquery += 1
            while model.predict_label(x0+torch.tensor(lbd_hi*theta, dtype=torch.float).cuda()) == y0:
                lbd_hi = lbd_hi*(1 + lin_search_ratio)
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*(1 - lin_search_ratio)
            nquery += 1
            while model.predict_label(x0+torch.tensor(lbd_lo*theta, dtype=torch.float).cuda()) != y0 :
                lbd_lo = lbd_lo*(1 - lin_search_ratio)
                nquery += 1
        
        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + torch.tensor(lbd_mid*theta, dtype=torch.float).cuda()) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery


    def fine_grained_binary_search(self, model, x0, y0, theta, initial_lbd, current_best, tol=1e-5):
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
        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + torch.tensor(lbd_mid*theta, dtype=torch.float).cuda()) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery


    def __call__(self, input_xi, label_or_target, targeted=False, distortion=None,
                 stopping=1e-8, query_limit=80000, args=None):
        self.args = args
        
        if targeted:
            raise NotImplementedError
            # adv = self.attack_targeted(input_xi, label_or_target, targeted, distortion=distortion,
            #                            svm=svm, stopping=stopping, query_limit=query_limit)
        else:
            adv = self.attack_untargeted(input_xi, label_or_target, distortion=distortion,
                                         stopping=stopping, query_limit=query_limit)
        return adv