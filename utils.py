def mulvt(v,t):
##################################
## widely used in binary search ##
## v is batch_size by 1         ##
## t is batch_size by any dim   ##
##################################
    batch_size, other_dim = t.size()[0], t.size()[1:]
    len_dim = len(other_dim)-1
    for i in range(len_dim):
        v = v.unsqueeze(len(v.size()))
    v = v.expand(t.size())
    return v*t    
    
def reduce_sum(t,axis):
    dim = t.size()
    
