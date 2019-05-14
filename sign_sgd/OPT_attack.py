from utils import mulvt
import time, torch
import numpy as np 
from numpy import linalg as LA

class OPT_attack(object):
    def __init__(self, model, train_dataset=None):
        self.model = model
        self.train_dataset = train_dataset

    def attack_untargeted(self, x0, y0, alpha = 0.2, beta = 0.001, iterations = 1500, query_limit=80000,
                          distortion=None, seed=None, stopping=0.0001):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """
        model = self.model
        y0 = y0[0]
        if (model.predict_label(x0) != y0):
            print("Fail to classify the image. No need to attack.")
            return x0, 0.0

        num_directions = 100
        best_theta, g_theta = None, float('inf')
        query_count = 0
        print("Searching for the initial direction on %d random directions: " % (num_directions))
        timestart = time.time()
        
        if seed is not None:
            np.random.seed(seed)
            
        for i in range(num_directions):
            query_count += 1
            theta = np.random.randn(*x0.shape)
            if model.predict_label(x0 + torch.tensor(theta, dtype=torch.float).cuda())!=y0:
                initial_lbd = LA.norm(theta)
                theta /= initial_lbd
                lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    print("--------> Found distortion %.4f" % g_theta)

        timeend = time.time()
        if g_theta == float('inf'):
            return "NA", float('inf')
        print("==========> Found best distortion %.4f in %.4f seconds using %d queries" %
              (g_theta, timeend-timestart, query_count))
    
        timestart = time.time()
        g1 = 1.0
        theta, g2 = best_theta, g_theta
        opt_count = 0
        prev_obj = 100000
        for i in range(iterations):
            gradient = np.zeros(theta.shape)
            q = 10
            min_g1 = float('inf')
            for _ in range(q):
                u = np.random.randn(*theta.shape)
                u /= LA.norm(u)
                ttt = theta+beta * u
                ttt /= LA.norm(ttt)
                g1, count = self.fine_grained_binary_search_local(model, x0, y0, ttt, initial_lbd=g2, tol=beta/500)
                opt_count += count
                if g1 == float('inf') or g2 == float('inf'):
                    continue
                gradient += (g1-g2)/beta * u
                if g1 < min_g1:
                    min_g1 = g1
                    min_ttt = ttt
            gradient = 1.0/q * gradient

            if opt_count > query_limit:
                break
                
            if distortion is not None:
                if g2 < distortion:
                    print("Success: required distortion reached")
                    break

            if (i+1)%10 == 0:
                print("Iteration %3d distortion %.4f num_queries %d" % (i+1, LA.norm(g2*theta), opt_count))
#                 if g2 > prev_obj-stopping:
#                     print("Stopping criteria reached")
#                     break
#                 prev_obj = g2

            min_theta = theta
            min_g2 = g2
        
            for _ in range(15):
                new_theta = theta - alpha * gradient
                new_theta /= LA.norm(new_theta)
                new_g2, count = self.fine_grained_binary_search_local(model, x0, y0, new_theta,
                                                                      initial_lbd=min_g2, tol=beta/500)
                opt_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                else:
                    break

            if min_g2 >= g2:
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = theta - alpha * gradient
                    new_theta /= LA.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local(model, x0, y0, new_theta,
                                                                          initial_lbd = min_g2, tol=beta/500)
                    opt_count += count
                    if new_g2 < g2:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        break

            if min_g2 <= min_g1:
                theta, g2 = min_theta, min_g2
            else:
                theta, g2 = min_ttt, min_g1

            if g2 < g_theta:
                best_theta, g_theta = theta, g2
            
            #print(alpha)
            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving, g2 %lf gtheta %lf beta is %lf" % (g2, g_theta, beta))
                beta = beta * 0.1
                if (beta < 1e-8):
                    break

        target = model.predict_label(x0 + torch.tensor(g_theta*best_theta, dtype=torch.float).cuda())
        timeend = time.time()
        print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" %
             (g_theta, target, query_count + opt_count, timeend-timestart))
        return x0 + torch.tensor(g_theta*best_theta, dtype=torch.float).cuda(), g_theta

    def fine_grained_binary_search_local(self, model, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
        tol = max(2e-8, tol)
         
        if model.predict_label(x0 + torch.tensor(lbd*theta, dtype=torch.float).cuda()) == y0:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while model.predict_label(x0 + torch.tensor(lbd_hi*theta, dtype=torch.float).cuda()) == y0:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 20:
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while model.predict_label(x0 + torch.tensor(lbd_lo*theta, dtype=torch.float).cuda()) != y0 :
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
            if model.predict_label(x0 + torch.tensor(current_best*theta, dtype=torch.float).cuda()) == y0:
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

    def attack_targeted(self, x0, y0, target, alpha=0.2, beta=0.001, iterations=1500, query_limit=80000,
                        distortion=None, seed=None, stopping=0.0001):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """
        model = self.model
        y0 = y0[0]
        print("Targeted attack - Source: {0} and Target: {1} Seed: {2}".format(y0, target.item(), seed))
        if (model.predict_label(x0) == target):
            print("Image already target. No need to attack.")
            return x0, 0.0

        if self.train_dataset is None:
            print("Need training dataset for initial theta.")
            return x0, float('inf')        

        if seed is not None:
            np.random.seed(seed)
        
        num_samples = 100
        best_theta, g_theta = None, float('inf')
        query_count = 0
        ls_total = 0
        sample_count = 0
        print("Searching for the initial direction on %d samples: " % (num_samples))
        timestart = time.time()

        # Iterate through training dataset. Find best initial point.
        for i, (xi, yi) in enumerate(self.train_dataset):
            yi_pred = model.predict_label(xi)
            query_count += 1
            if yi_pred != target:
                continue
            
            theta = xi.cpu().numpy() - x0.cpu().numpy()
            initial_lbd = LA.norm(theta)
            theta /= initial_lbd
            lbd, count = self.fine_grained_binary_search_targeted(model, x0, y0, target, theta, initial_lbd, g_theta)
            query_count += count
            if lbd < g_theta:
                best_theta, g_theta = theta, lbd
                print("--------> Found distortion %.4f" % g_theta)

            sample_count += 1
            if sample_count >= num_samples:
                break
                
            if i > 500:
                break

        timeend = time.time()
        if g_theta == float('inf'):
            return x0, float('inf')
        print("==========> Found best distortion %.4f in %.4f seconds using %d queries" %
              (g_theta, timeend-timestart, query_count))
    
        timestart = time.time()
        g1 = 1.0
        theta, g2 = best_theta, g_theta
        opt_count = 0
        prev_obj = 100000
        for i in range(iterations):
            gradient = np.zeros(theta.shape)
            q = 10
            min_g1 = float('inf')
            for _ in range(q):
                u = np.random.randn(*theta.shape)
                u /= LA.norm(u)
                ttt = theta+beta * u
                ttt /= LA.norm(ttt)
                g1, count = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, ttt,
                                                                           initial_lbd=g2, tol=beta/500)
                opt_count += count
                if g1 == float('inf') or g2 == float('inf'):
                    continue
                gradient += (g1-g2)/beta * u
                if g1 < min_g1:
                    min_g1 = g1
                    min_ttt = ttt
            gradient = 1.0/q * gradient

            if opt_count > query_limit:
                break
                
            if distortion is not None:
                if g2 < distortion:
                    print("Success: required distortion reached")
                    break

            if (i+1)%10 == 0:
                print("Iteration %3d distortion %.4f num_queries %d" % (i+1, LA.norm(g2*theta), opt_count))
#                 if g2 > prev_obj-stopping:
#                     print("Stopping criteria reached")
#                     break

                prev_obj = g2

            min_theta = theta
            min_g2 = g2
        
            for _ in range(15):
                new_theta = theta - alpha * gradient
                new_theta /= LA.norm(new_theta)
                new_g2, count = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, new_theta,
                                                                               initial_lbd=min_g2, tol=beta/500)
                opt_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                else:
                    break

            if min_g2 >= g2:
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = theta - alpha * gradient
                    new_theta /= LA.norm(new_theta)
                    new_g2, count = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, new_theta,
                                                                                   initial_lbd=min_g2, tol=beta/500)
                    opt_count += count
                    if new_g2 < g2:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        break

            if min_g2 <= min_g1:
                theta, g2 = min_theta, min_g2
            else:
                theta, g2 = min_ttt, min_g1

            if g2 < g_theta:
                best_theta, g_theta = theta, g2
            
            #print(alpha)
            if alpha < 1e-4:
                alpha = 1.0
                print("Warning: not moving, g2 %lf gtheta %lf beta is %lf" % (g2, g_theta, beta))
                beta = beta * 0.1
                if (beta < 1e-8):
                    break
                
        target = model.predict_label(x0 + torch.tensor(g_theta*best_theta, dtype=torch.float).cuda())
        timeend = time.time()
        print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" %
             (g_theta, target, query_count + opt_count, timeend-timestart))
        return x0 + torch.tensor(g_theta*best_theta, dtype=torch.float).cuda(), g_theta
    
    def fine_grained_binary_search_local_targeted(self, model, x0, y0, t, theta, initial_lbd=1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
        tol = max(2e-8, tol)

        if model.predict_label(x0 + torch.tensor(lbd*theta, dtype=torch.float).cuda()) != t:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while model.predict_label(x0 + torch.tensor(lbd_hi*theta, dtype=torch.float).cuda()) != t:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 100: 
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while model.predict_label(x0 + torch.tensor(lbd_lo*theta, dtype=torch.float).cuda()) == t:
                lbd_lo = lbd_lo*0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + torch.tensor(lbd_mid*theta, dtype=torch.float).cuda()) == t:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid

#         temp_theta = np.abs(lbd_hi*theta)
#         temp_theta = np.clip(temp_theta - 0.15, 0.0, None)
#         loss = np.sum(np.square(temp_theta))
        return lbd_hi, nquery

    def fine_grained_binary_search_targeted(self, model, x0, y0, t, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best: 
            if model.predict_label(x0 + torch.tensor(current_best*theta, dtype=torch.float).cuda()) != t:
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
            if model.predict_label(x0 + torch.tensor(lbd_mid*theta, dtype=torch.float).cuda()) != t:
                lbd_lo = lbd_mid
            else:
                lbd_hi = lbd_mid
        return lbd_hi, nquery

    def __call__(self, input_xi, label_or_target, target=None, distortion=None, seed=None,
                 query_limit=80000, stopping=0.0001):
        if target is not None:
            adv = self.attack_targeted(input_xi, label_or_target, target, distortion=distortion,
                                       seed=seed, query_limit=query_limit, stopping=stopping)
        else:
            adv = self.attack_untargeted(input_xi, label_or_target, distortion=distortion,
                                         seed=seed, query_limit=query_limit, stopping=stopping)
        return adv   
    
        
