import torch
import scipy

class Evolutionary(object):
    def __init__(self,model):
        self.model = model

    def loss(self, x0, y0, x_):
        if self.model.predict_label(torch.tensor(x_, dtype=torch.float).cuda())!=torch.tensor(y0):
            return np.inf
        else:
            return np.linalg.norm(x - x_)
        
    def evolutionary(self, x0, y0, TARGETED=False):    
        num_directions = 100
        best_dir, best_dist = None, float('inf')
        
        if not isinstance(image, np.ndarray):
            x0 = x0.cpu().numpy()
            y0 = y0.cpu().numpy()
            
        print("Searching for the initial direction on %d random directions: " % (num_directions))
        
        for i in range(num_directions):
            query_count += 1
            theta = np.random.randn(x0.shape)
            if self.model.predict_label(torch.tensor(x0+theta, dtype=torch.float).cuda())!= torch.tensor(y0):
                if best_dist < np.linalg.norm(theta):
                    best_dir, best_dist = theta, np.linalg.norm(theta)
                    print("--------> Found distortion %.4f" % best_dist)

        print("==========> Found best distortion %.4f "
              "using %d queries" % (best_dist, query_count))

        n_shape = (32, 32, 3)

        # Hyperparameters
        sigma = 0.01
        cc = 0.01
        cconv = 0.001
        m = 15 * 15 * 3
        m_shape = (15, 15, 3)
        k = int(m/20)
        mu = 0.1
        
        # Initializations
        C = np.identity(m)
        x_ = x0 + theta
        pc = np.zeros(m)
        
        prev_loss = best_dist
        
        for it in rang(1000):
            z = np.random.multivariate_normal(np.zeros(m), (sigma**2)*C)
            
            # Select k coordinates with probability proportional to C
            probs = np.exp(np.diagonal(C))
            probs /= sum(probs)
            indices = np.random.choice(m, size=k, replace=False, p=probs)
            
            # Set non selected coordinates to 0
            indices_zero = np.setdiff1d(np.arange(m), indices)
            z[indices_zero] = 0
            
            # Bilinear upsampling
            z = np.reshape(z, m_shape)
            z_ = scipy.misc.imresize(z, n_shape, interp='bilinear')
            z_ = z_ + mu*(x - x_)
            
            query_count += 1
            new_loss = self.loss(x_ + z_)
            if  new_loss < prev_loss:
                # Update x_
                x_ = x_ + z_
                
                # Update pc and C
                pc = (1-cc)*pc + z*np.sqrt(cc*(2-cc))/sigma

                c_diag = np.diagonal(C)
                c_diag = (1-cconv)*c_diag + cconv*np.square(pc)
                C = np.diag(c_diag)
                
                # Update loss
                prev_loss = new_loss
        
        return x_


    def __call__(self, input_xi, target, TARGETED=True):
        adv = self.evolutionary(input_xi, target , TARGETED)
        return adv
    
        
