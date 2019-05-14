import torch
import numpy as np
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
    
    def predict_label(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).type(torch.FloatTensor)
        image = torch.clamp(image,self.bounds[0],self.bounds[1]).cuda()
        if len(image.size())!=4:
            image = image.unsqueeze(0)
        with torch.no_grad():
            image = Variable(image) # ?? not supported by latest pytorch
        output = self.model(image)
        #image = Variable(image, volatile=True) # ?? not supported by latest pytorch
        _, predict = torch.max(output.data, 1)
        return predict[0]

    def predict_batch_label(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).type(torch.FloatTensor)
        image = torch.clamp(image,self.bounds[0],self.bounds[1]).cuda()
        if len(image.size())!=4:
            image = image.unsqueeze(0)
        with torch.no_grad():
            image = Variable(image) # ?? not supported by latest pytorch
        output = self.model(image)
        _, predict = torch.max(output.data, 1)
        return predict
        

    def get_gradient(self,loss):
        loss.backward()
        
