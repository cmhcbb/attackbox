import torch
from torch.autograd import Variable

class PytorchModel(object):
    def __init__(self,model, bounds, num_classes):
        self.model = model
        self.model.eval()
        self.bounds = bounds
        self.num_classes = num_classes
    
    def predict(self,image):
        image = torch.clamp(image,self.bounds[0],self.bounds[1]).cuda()
        if len(image.size())!=4:
            image = image.unsqueeze(0)
        #image = Variable(image, volatile=True) # ?? not supported by latest pytorch
        output = self.model(image)
        return output

    def get_gradient(self,loss):
        loss.backward()
        
