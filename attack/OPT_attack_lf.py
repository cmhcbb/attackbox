import time
import numpy as np 
from numpy import linalg as LA
import random

class OPT_attack_lf(object):
    def __init__(self,model):
        self.model = model

    def attack_untargeted(self, x0, y0, alpha = 0.2, beta = 0.01, iterations = 1000):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """
        model = self.model
        y0 = y0[0]
        if (model.predict_label(x0) != y0):
            print("Fail to classify the image. No need to attack.")
            return x0,0,0

        num_directions = 10
        best_theta, g_theta = None, float('inf')
        query_count = 0
        print("Searching for the initial direction on %d random directions: " % (num_directions))
        timestart = time.time()
        for i in range(num_directions):
            query_count += 1
            theta = np.random.randn(*x0.shape)
            if model.predict_label(x0+theta)!=y0:
                #l2norm = LA.norm(theta)
                initial_lbd = LA.norm(theta.flatten(),np.inf)
                theta /= initial_lbd     # might have problem on the defination of direction
                lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
                query_count += count
                if lbd < g_theta:
                    best_theta, g_theta = theta, lbd
                    print("--------> Found distortion %.4f" % g_theta)

        timeend = time.time()
        if g_theta == np.inf:
            return "NA", float('inf'), 0
        print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))
    
        timestart = time.time()
        g1 = 1.0
        theta, g2 = best_theta, g_theta
        opt_count = 0
        stopping = 0.005
        prev_obj = 100000
        for i in range(iterations):
            gradient = np.zeros(theta.shape)
            q = 5
            min_g1 = float('inf')
            for _ in range(q):
                u = np.random.randn(*theta.shape)
                u /= LA.norm(u.flatten(),np.inf)
                ttt = theta+beta * u
                ttt /= LA.norm(ttt.flatten(),np.inf)
                g1, count = self.fine_grained_binary_search_local(model, x0, y0, ttt, initial_lbd = g2, tol=beta/500)
                opt_count += count
                gradient += (g1-g2)/beta * u
                if g1 < min_g1:
                    min_g1 = g1
                    min_ttt = ttt
            gradient = 1.0/q * gradient

            if (i+1)%1 == 0:
                print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (i+1, g1, g2, LA.norm((g2*theta).flatten(),np.inf), opt_count))
                if g2 > prev_obj-stopping:
                    print("stopping")
                    break
                prev_obj = g2

            min_theta = theta
            min_g2 = g2
        
            for _ in range(15):
                new_theta = theta - alpha * gradient
                new_theta /= LA.norm(new_theta.flatten(),np.inf)
                new_g2, count = self.fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
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
                    new_theta /= LA.norm(new_theta.flatten(),np.inf)
                    new_g2, count = self.fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
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
                print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
                beta = beta * 0.1
                if (beta < 0.00005):
                    break

        target = model.predict_label(x0 + g_theta*best_theta)
        timeend = time.time()
        print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
        return x0 + g_theta*best_theta, g_theta, query_count + opt_count

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

    def fine_grained_binary_search(self, model, x0, y0, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best: 
            if model.predict_label(x0+current_best*theta) == y0:
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
            if model.predict_label(x0 + lbd_mid*theta) != y0:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def attack_targeted(self, initial_xi, x0, y0, target, alpha = 0.2, beta = 0.001, iterations = 5000):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """
        model = self.model
        #print(y0)
        #y0 = y0[0]
        if (model.predict_label(x0) != y0):
            print("Fail to classify the image. No need to attack.")
            return x0,0,0

        num_samples = 100
        best_theta, g_theta = None, float('inf')
        query_count = 0
        sample_count = 0
        #print("Searching for the initial direction on %d samples: " % (num_samples))
        timestart = time.time()
        #samples = set(random.sample(range(len(train_dataset)), num_samples))
        #train_dataset = train_dataset[samples]
        #for i, (xi, yi) in enumerate(train_dataset):
        #    if i not in samples:
        #        continue   
        #    if yi != target:
        #        continue
        #    query_count += 1
        #    if model.predict(xi) == target:
        #       theta = xi - x0
                #l2norm = LA.norm(theta)
        #        initial_lbd = LA.norm(theta.flatten(),np.inf)
        #        theta /= initial_lbd     # might have problem on the defination of direction
        #        lbd, count = self.fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
        #       query_count += count
        #        if lbd < g_theta:
        #            best_theta, g_theta = theta, lbd
        #           print("--------> Found distortion %.4f" % g_theta)
        #print(x0)
        xi = initial_xi
        xi = xi.numpy()
        theta = xi - x0
        initial_lbd = LA.norm(theta.flatten(),np.inf)
        theta /= initial_lbd     # might have problem on the defination of direction
        lbd, count, lbd_g2 = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, theta)
        query_count += count
        if lbd < g_theta:
            best_theta, g_theta = theta, lbd
            print("--------> Found distortion %.4f" % g_theta)        

        timeend = time.time()
        if g_theta == np.inf:
            return "NA", float('inf'), 0
        print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))
    
        timestart = time.time()
        g1 = 1.0
        theta, g2 = best_theta, g_theta
        opt_count = 0
        stopping = 1e-8
        prev_obj = 1000000
        for i in range(iterations):
            if g2==0.0:
                break
            gradient = np.zeros(theta.shape)
            q = 20
            min_g1 = float('inf') 
            min_lbd = float('inf')
            for _ in range(q):
                u = np.random.randn(*theta.shape)
                u /= LA.norm(u.flatten(),np.inf)
                ttt = theta+beta * u
                ttt /= LA.norm(ttt.flatten(),np.inf)
                g1, count, lbd_hi = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, ttt, initial_lbd = lbd_g2, tol=beta/500)
                #g1, count, lbd_hi = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, ttt)
                opt_count += count
                gradient += (g1-g2)/beta * u
                if g1 < min_g1:
                    min_g1 = g1
                    min_ttt = ttt
                    min_lbd_1 = lbd_hi
            gradient = 1.0/q * gradient

            if (i+1)%10 == 0:
                print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (i+1, g1, g2, LA.norm((lbd_g2*theta).flatten(),np.inf), opt_count))
                if g2 > prev_obj-stopping:
                    print("stopping")
                    break
                prev_obj = g2

            min_theta = theta
            min_g2 = g2
            min_lbd = lbd_g2

            for _ in range(15):
                new_theta = theta - alpha * gradient
                new_theta /= LA.norm(new_theta.flatten(),np.inf)
                new_g2, count, lbd_hi = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, new_theta, initial_lbd = min_lbd, tol=beta/500)
                #new_g2, count, lbd_hi = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, new_theta)
                opt_count += count
                alpha = alpha * 2
                if new_g2 < min_g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                    min_lbd = lbd_hi
                else:
                    break


            if min_g2 >= g2:
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_theta = theta - alpha * gradient
                    new_theta /= LA.norm(new_theta.flatten(),np.inf)
                    new_g2, count, lbd_hi = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, new_theta, initial_lbd = min_lbd, tol=beta/500)
                    #new_g2, count, lbd_hi = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, new_theta)
                    opt_count += count
                    if new_g2 < g2:
                        min_theta = new_theta 
                        min_g2 = new_g2
                        min_lbd = lbd_hi
                        break

            if min_g2 <= min_g1:
                theta, g2 = min_theta, min_g2
                lbd_g2 = min_lbd
            else:
                theta, g2 = min_ttt, min_g1
                lbd_g2 = min_lbd_1
            if g2 < g_theta:
                best_theta, g_theta = theta, g2
                #lbd_g2 = min_lbd
            #print(alpha)
            if alpha < 1e-6:
                alpha = 1.0
                print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
                beta = beta * 0.1
                if (beta < 1e-8):
                    break
        g_theta, _ = self.fine_grained_binary_search_local_targeted_original(model, x0, y0, target, best_theta, initial_lbd = 1.0, tol=beta/500)
        dis = LA.norm((g_theta*best_theta).flatten(),np.inf)
        target = model.predict_label(x0 + g_theta*best_theta)
        timeend = time.time()
        print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (dis, target, query_count + opt_count, timeend-timestart))
        return x0 + g_theta*best_theta

    def fine_grained_binary_search_local_targeted(self, model, x0, y0, t, theta, initial_lbd= 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd

        if model.predict_label(x0+lbd*theta) != t:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while model.predict_label(x0+lbd_hi*theta) != t:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 100: 
                    return float('inf'), nquery, 1.0
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while model.predict_label(x0+lbd_lo*theta) == t:
                lbd_lo = lbd_lo*0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + lbd_mid*theta) == t:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        temp_theta = np.abs(lbd_hi*theta)
        temp_theta = np.clip(temp_theta - 0.15, 0.0, None)
        loss = np.sum(np.square(temp_theta))
        #print(lbd_hi)
        return loss, nquery, lbd_hi

    def fine_grained_binary_search_local_targeted_original(self, model, x0, y0, t, theta, initial_lbd = 1.0, tol=1e-5):
        nquery = 0
        lbd = initial_lbd
       
        if model.predict_label(x0+lbd*theta) != t:
            lbd_lo = lbd
            lbd_hi = lbd*1.01
            nquery += 1
            while model.predict_label(x0+lbd_hi*theta) != t:
                lbd_hi = lbd_hi*1.01
                nquery += 1
                if lbd_hi > 100: 
                    return float('inf'), nquery
        else:
            lbd_hi = lbd
            lbd_lo = lbd*0.99
            nquery += 1
            while model.predict_label(x0+lbd_lo*theta) == t:
                lbd_lo = lbd_lo*0.99
                nquery += 1

        while (lbd_hi - lbd_lo) > tol:
            lbd_mid = (lbd_lo + lbd_hi)/2.0
            nquery += 1
            if model.predict_label(x0 + lbd_mid*theta) == t:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def fine_grained_binary_search_targeted(self, model, x0, y0, t, theta, initial_lbd, current_best):
        nquery = 0
        if initial_lbd > current_best: 
            if model.predict_label(x0+current_best*theta) != t:
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
            if model.predict_label(x0 + lbd_mid*theta) == t:
                lbd_hi = lbd_mid
            else:
                lbd_lo = lbd_mid
        return lbd_hi, nquery

    def __call__(self, input_xi, label_or_target, initial_xi=None, target=None, TARGETED=False):
        if TARGETED:
            adv = self.attack_targeted(initial_xi, input_xi, label_or_target, target)
        else:
            adv = self.attack_untargeted(input_xi, label_or_target)
        return adv   
    
        
