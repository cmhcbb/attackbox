import torch
import scipy
import numpy as np
import scipy.misc
import PIL

class Evolutionary(object):
    def __init__(self,model, train_dataset=None):
        self.model = model

    def predict(self, x):
        return self.model.predict_label(torch.tensor(x.reshape(3, 32, 32), dtype=torch.float))
    
    def loss(self, x0, y0, x_):
        if self.predict(x_) == y0:
            return np.inf
        else:
            return np.linalg.norm(x0 - x_)
        
    def evolutionary(self, x0, y0, TARGETED=False):    
        num_directions = 100
        best_dir, best_dist = None, float('inf')
        query_count = 0 
        if not isinstance(x0, np.ndarray):
            x0 = x0.cpu().numpy()
            y0 = y0.cpu().numpy()

        n_shape = (32, 32, 3)
            
        x0 = x0[0].reshape(n_shape)
        y0 = y0[0]
        print("Searching for the initial direction on %d random directions: " % (num_directions))
        
        for i in range(num_directions):
            query_count += 1
            theta = np.random.randn(*x0.shape)
            if self.predict(x0+theta)!= y0:
                if np.linalg.norm(theta) < best_dist:
                    best_dir, best_dist = theta, np.linalg.norm(theta)
                    print("--------> Found distortion %.4f" % best_dist)

        print("==========> Found best distortion %.4f "
              "using %d queries" % (best_dist, query_count))

        # Hyperparameters
        #sigma = 0.01
        cc = 0.01
        cconv = 0.001
        m = 16 * 16 * 3
        m_shape = (16, 16, 3)
        k = int(m/20)
        mu = 0.1
        
        # Hyperparameter tuning - 1/5 success rule
        MAX_PAST_TRIALS = 25
        success_past_trials = 0
        
        # Initializations
        C = np.identity(m)
        x_ = x0 + theta
        pc = np.zeros(m)
        
        prev_loss = best_dist
        
        for it in range(10000):
            if it%100 == 0:
                print("Iteration: ", it, " mu: ", mu)

            # Update hyperparameters
            if it > MAX_PAST_TRIALS and it%5==0:
                p = success_past_trials/MAX_PAST_TRIALS
                mu = mu*np.exp(p - 0.2)
                #print("p ", p, "mu ", mu)
            sigma = 0.01 * np.linalg.norm(x_ - x0)
            
            z = np.random.multivariate_normal(np.zeros(m), (sigma**2)*C)
            
            
            # Select k coordinates with probability proportional to C
            probs = np.exp(np.diagonal(C))
            probs /= sum(probs)
            indices = np.random.choice(m, size=k, replace=False, p=probs)
            
            # Set non selected coordinates to 0
            indices_zero = np.setdiff1d(np.arange(m), indices)
            z[indices_zero] = 0
            
            # Bilinear upsampling
            #z = np.reshape(z, m_shape)
            z_ = scipy.ndimage.zoom(z.reshape(m_shape), [2, 2, 1], order=1)
            #z_ = scipy.misc.imresize(z, n_shape, interp='bilinear', mode='P')
            z_ = z_ + mu*(x0 - x_)
            
            query_count += 1
            new_loss = self.loss(x0, y0, x_ + z_)
            success = new_loss < prev_loss
            
            if  success:
                # Update x_
                x_ = x_ + z_
                print("Found adv with distortion {0} Queries {1}".format(np.linalg.norm(x_ - x0), query_count))
                
                # Update pc and C
                pc = (1-cc)*pc + z*np.sqrt(cc*(2-cc))/sigma

                c_diag = np.diagonal(C)
                c_diag = (1-cconv)*c_diag + cconv*np.square(pc)
                C = np.diag(c_diag)
                
                # Update loss
                prev_loss = new_loss
            
            # Update past success trials.
            if success:
                success_past_trials += 1
            else:
                success_past_trials -= 1
            success_past_trials = np.clip(success_past_trials, 0, MAX_PAST_TRIALS)

        return x_


    def __call__(self, input_xi, yi, TARGETED=True, seed=None, distortion=None, query_limit=None, stopping=None):
        adv = self.evolutionary(input_xi, yi)
        return adv
    
        
