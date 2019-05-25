import time
import random 
import numpy as np
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import os
from PIL import Image
#import pretrainedmodels
#import pretrainedmodels.utils as utils
import torchvision.models as models
import torch.nn.functional as F
from layers.feat_noise import Noise


# Hyperparameters
num_epochs = 50
batch_size = 128
learning_rate = 0.001

class IMAGENET():
    def __init__(self, arch):
        self.model = models.__dict__[arch](pretrained=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model, device_ids=[0])
 
    def predict(self, image):
        #image = torch.clamp(image, -1, 1)
        if len(image.size())==3:
            image[0] = torch.clamp(image[0], min=-0.485/0.229, max= (1-0.485)/0.229)
            image[1] = torch.clamp(image[1], min=-0.456/0.224, max= (1-0.456)/0.224)
            image[2] = torch.clamp(image[2], min = -0.406/0.225, max= (1-0.406)/0.225)
        else:
            image = image[0]
            image[0] = torch.clamp(image[0], min=-0.485/0.229, max= (1-0.485)/0.229)
            image[1] = torch.clamp(image[1], min=-0.456/0.224, max= (1-0.456)/0.224)
            image[2] = torch.clamp(image[2], min = -0.406/0.225, max= (1-0.406)/0.225)
            
        image = Variable(image, volatile=True).view(1,3,224,224)
        output = self.model(image)
        _, predict = torch.max(output.data, 1)
        return predict[0]
    
    def predict_batch(self, image):
        #image = torch.clamp(image, -1 ,1)
        #image[:,0] = torch.clamp(image[:,0], min=-0.485/0.229, max= (1-0.485)/0.229)
        #print(image[:,0])
        #image[:,1] = torch.clamp(image[:,1], min=-0.456/0.224, max= (1-0.456)/0.224)
        #image[:,2] = torch.clamp(image[:,2], min = -0.406/0.225, max= (1-0.406)/0.225)
        image = Variable(image, volatile=True)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self.model(image)
        _, predict = torch.max(output.data, 1)
        return predict

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_plain(nn.Module):
    def __init__(self, vgg_name, nclass, img_width=32):
        super(VGG_plain, self).__init__()
        self.img_width = img_width
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, nclass)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        width = self.img_width
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                width = width // 2
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=width, stride=1)]
        return nn.Sequential(*layers)

    def predict(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        image = Variable(image, volatile=True).view(1,3, 32,32)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)[0]
        print(output)
        _, predict = torch.max(output.data, 1)
        return predict[0]
    
    def predict_batch(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        image = Variable(image, volatile=True)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)[0]
        _, predict = torch.max(output.data, 1)
        return predict



class VGG_rse(nn.Module):
    def __init__(self, vgg_name, nclass, noise_init, noise_inner, img_width=32):
        super(VGG_rse, self).__init__()
        self.noise_init = noise_init
        self.noise_inner = noise_inner
        self.img_width = img_width
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, nclass)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        width = self.img_width
        for i, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                width = width // 2
            else:
                if i == 0:
                    noise_layer = Noise(self.noise_init)
                else:
                    noise_layer = Noise(self.noise_inner)
                layers += [noise_layer,
                           nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=width, stride=1)]
        return nn.Sequential(*layers)

    def predict(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        image = Variable(image, volatile=True).view(1,3, 32,32)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)[0]
        _, predict = torch.max(output.data, 1)
        return predict[0]
    
    def predict_batch(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        image = Variable(image, volatile=True)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)[0]
        _, predict = torch.max(output.data, 1)
        return predict


class VGG_vi(nn.Module):
    def __init__(self, sigma_0, N, init_s, vgg_name, nclass, img_width=32):
        super(VGG_vi, self).__init__()
        self.sigma_0 = sigma_0
        self.N = N
        self.init_s = init_s
        self.img_width = img_width
        self.classifier = RandLinear(sigma_0, N, init_s, 512, nclass)
        self.features = self._make_layers(cfg[vgg_name])

    def forward(self, x):
        kl_sum = 0
        out = x
        for l in self.features:
            if type(l).__name__.startswith("Rand"):
                out, kl = l.forward(out)
                if kl is not None:
                    kl_sum += kl
            else:
                out = l.forward(out)
        out = out.view(out.size(0), -1)
        out, kl = self.classifier.forward(out)
        kl_sum += kl
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        width = self.img_width
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                width = width // 2
            else:
                layers += [
                        RandConv2d(self.sigma_0, self.N, self.init_s, in_channels, x, kernel_size=3, padding=1),
                        RandBatchNorm2d(self.sigma_0, self.N, self.init_s, x),
                        nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=width, stride=1)]
        return nn.Sequential(*layers)
    
    def predict(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        image = Variable(image, volatile=True).view(1,3, 32,32)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)[0]
        _, predict = torch.max(output.data, 1)
        return predict[0]
    
    def predict_batch(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        image = Variable(image, volatile=True)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)[0]
        _, predict = torch.max(output.data, 1)
        return predict


class CIFAR10(nn.Module):
    def __init__(self):
        super(CIFAR10, self).__init__()
        self.features = self._make_layers()
        self.fc1 = nn.Linear(3200,256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256,256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256,10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def _make_layers(self):
        layers=[]
        in_channels= 3
        layers += [nn.Conv2d(in_channels, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.Conv2d(64, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(64, 128, kernel_size=3),
                   nn.BatchNorm2d(128),
                   nn.ReLU()]
        layers += [nn.Conv2d(128, 128, kernel_size=3),
                   nn.BatchNorm2d(128),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        return nn.Sequential(*layers)


    def predict(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        image = Variable(image, volatile=True).view(1,3, 32,32)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict[0]
    
    def predict_batch(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        image = Variable(image, volatile=True)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict



class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.features = self._make_layers()
        self.fc1 = nn.Linear(1024,200)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(200,200)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(200,10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out

    def _make_layers(self):
        layers=[]
        in_channels= 1
        layers += [nn.Conv2d(in_channels, 32, kernel_size=3),
                   nn.BatchNorm2d(32),
                   nn.ReLU()]
        layers += [nn.Conv2d(32, 32, kernel_size=3),
                   nn.BatchNorm2d(32),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.Conv2d(32, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.Conv2d(64, 64, kernel_size=3),
                   nn.BatchNorm2d(64),
                   nn.ReLU()]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        return nn.Sequential(*layers)


    def predict(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        image = Variable(image, volatile=True).view(1,1,28,28)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict[0]

    def predict_batch(self, image):
        self.eval()
        image = torch.clamp(image,0,1)
        image = Variable(image, volatile=True)
        if torch.cuda.is_available():
            image = image.cuda()
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict



class SimpleMNIST(nn.Module):
    """ Custom CNN for MNIST
        stride = 1, padding = 2
        Layer 1: Conv2d 5x5x16, BatchNorm(16), ReLU, Max Pooling 2x2
        Layer 2: Conv2d 5x5x32, BatchNorm(32), ReLU, Max Pooling 2x2
        FC 10
    """
    def __init__(self):
        super(SimpleMNIST, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )
        self.fc = nn.Linear(7*7*32, 10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def predict(self, image):
        self.eval()
        image = Variable(image.unsqueeze(0))
        output = self(image)
        _, predict = torch.max(output.data, 1)
        return predict[0]


def show_image(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))

def load_mnist_data(test_batch_size=1):
    """ Load MNIST data from torchvision.datasets 
        input: None
        output: minibatches of train and test sets 
    """
    # MNIST Dataset
    train_dataset = dsets.MNIST(root='./data/mnist', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.MNIST(root='./data/mnist', train=False, transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset

def load_cifar10_data(test_batch_size=1):
    """ Load MNIST data from torchvision.datasets 
        input: None
        output: minibatches of train and test sets 
    """
    # CIFAR10 Dataset
    train_dataset = dsets.CIFAR10('./data/cifar10-py', download=True, train=True, transform= transforms.ToTensor())
    test_dataset = dsets.CIFAR10('./data/cifar10-py', download=True, train=False, transform= transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset, test_dataset

def load_imagenet_data():
    """ Load MNIST data from torchvision.datasets 
        input: None
        output: minibatches of train and test sets 
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])                         
    # train_dataset = dsets.ImageFolder(
    #     '/data/train',
    #     transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    val_dataset = dsets.ImageFolder(
        '/data3/ILSVRC2012/val/',
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

    # Data Loader (Input Pipeline)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000, shuffle=True)

    return val_loader, val_loader, val_dataset, val_dataset


class ImagenetTestDataset(Dataset):
    def __init__(self, root_file, transform=None):
       self.label =[]
       self.root_dir = root_file
       self.transform = transform
       self.img_name = sorted(os.listdir(root_file))
       for img in self.img_name:
           name = img.split('.')
           self.label.append(int(name[0])-1)

    def __getitem__(self, idx):
        image = Image.open(self.root_dir + '/' + self.img_name[idx])
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        #label = torch.LongTensor(self.label[idx])
        label = self.label[idx] 
        return image, label

    def __len__(self):
        return len(self.label)

def imagenettest():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    #test_dataset = ImagenetTestDataset('/data/test')

    test_dataset = ImagenetTestDataset('/data3/val', transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize,]))

    # Data Loader (Input Pipeline)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10, shuffle=True)

    return test_loader, test_dataset



def train_simple_mnist(model, train_loader):
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            labels = Variable(labels)
        
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d] Loss: %.4f' 
                    %(epoch+1, num_epochs, i+1, loss.data[0]))


def train_mnist(model, train_loader):
    # Loss and Optimizer
    model.train()
    lr = 0.01
    momentum = 0.9
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    # Train the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            images = Variable(images)
            labels = Variable(labels)
        
            # Forward + Backward + Optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d] Loss: %.4f' 
                    %(epoch+1, num_epochs, i+1, loss.data[0]))


def test_mnist(model, test_loader):
    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy of the model on the 10000 test images: %.2f %%' % (100.0 * correct / total))

def cross_entropy(log_input, target):
    product = log_input * target
    loss = torch.sum(product)
    loss *= -1/log_input.size()[0]
    return loss


def train_teacher(teacher, train_loader, test_loader, temp):
    teacher.train()
    lr = 0.1
    momentum = 0.9
    m = nn.LogSoftmax(dim=1)
    nllloss = nn.NLLLoss()
    optimizer = torch.optim.SGD(teacher.parameters(), lr=lr, momentum=momentum, nesterov=True)
    # Train the Teacher
    for epoch in range(num_epochs):
        '''
        if epoch%10==0 and epoch!=0:
            lr = lr * 0.95
            momentum = momentum * 0.95
            optimizer = torch.optim.SGD(teacher.parameters(), lr=lr, momentum=momentum, nesterov=True)
        '''
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            images = Variable(images)
            labels = Variable(labels)
        
            # Forward + Backward + Optimize
            outputs = teacher(images)           
            loss = nllloss(m(outputs/temp), labels)
            loss.backward()
            optimizer.step()
        
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d] Loss: %.4f' 
                    %(epoch+1, num_epochs, i+1, loss.data[0]))
    file_n = './models/dd_mnist_teacher'+str(temp) + '.pt'
    save_model(teacher,file_n)
    test_mnist(teacher,test_loader)

def train_student(student, teacher, train_loader, test_loader, temp):
    m = nn.Softmax(dim=1)
    nm= nn.LogSoftmax(dim=1)
    nllloss = nn.NLLLoss()
    teacher.eval()
    '''
    labels_a = []
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images  = images.cuda()
        images_v = Variable(images, volatile=True)
        output = teacher(images_v)/temp
        #print(output)
        labels = m(output)
        #print(labels)
        labels_a.append(labels.data)
    '''
    print("--------Training student------")
    lr = 0.1
    momentum = 0.9
    student.train()
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=momentum, nesterov=True)
    # Train the Student
    for epoch in range(num_epochs):
        '''
        if epoch%10==0 and epoch!=0:
            lr = lr * 0.95
            momentum = momentum * 0.95
            optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=momentum, nesterov=True)
        '''
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                images  = images.cuda()
            #labels = Variable(labels.cuda())
            images = Variable(images, volatile=False)
            # Forward + Backward + Optimize
            label_t = m(teacher(images)/temp)
            #print(label_t)
            outputs = student(images)     
            labels = Variable(label_t.data)
            #print(outputs)
            loss = cross_entropy(nm(outputs/temp), labels)
            loss.backward()
            optimizer.step()
        
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d] Loss: %.4f' 
                    %(epoch+1, num_epochs, i+1, loss.data[0]))

    file_n = './models/dd_mnist_student'+str(temp) + '.pt'
    save_model(student,file_n)
    test_mnist(student,test_loader)
         

def train_cifar10(model, train_loader):
    # Loss and Optimizer
    model.train()
    lr = 0.01
    momentum = 0.9
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    # Train the Model
    for epoch in range(num_epochs):
        if epoch%10==0 and epoch!=0:
            lr = lr * 0.95
            momentum = momentum * 0.95
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            images = Variable(images)
            labels = Variable(labels)
        
            # Forward + Backward + Optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Iter [%d] Loss: %.4f' 
                    %(epoch+1, num_epochs, i+1, loss.data[0]))
    return model

def test_cifar10(model, test_loader):
    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0
    total = 0
    for images, labels in test_loader:
        if torch.cuda.is_available():
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Test Accuracy of the model on the 10000 test images: %.4f %%' % (100.0 * correct / total))

class ToSpaceBGR(object):
    def __init__(self, is_bgr):
        self.is_bgr = is_bgr
    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor

class ToRange255(object):
    def __init__(self, is_255):
        self.is_255 = is_255
    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor

def save_model(model, filename):
    """ Save the trained model """
    torch.save(model.state_dict(), filename)

def load_model(model, filename):
    """ Load the training model """
    model.load_state_dict(torch.load(filename))

if __name__ == '__main__':
    train_loader, test_loader, train_dataset, test_dataset = load_mnist_data()
    teacher = MNIST()
    student = MNIST()
    #train_loader, test_loader, train_dataset, test_dataset = load_cifar10_data()
    #teacher = CIFAR10()
    #student = CIFAR10()
    #net = CIFAR10()
    if torch.cuda.is_available():
    #    net.cuda()
    #    net = torch.nn.DataParallel(net, device_ids=[0])
        teacher.cuda()
        student.cuda()
        teacher = torch.nn.DataParallel(teacher, device_ids=[0])
        student = torch.nn.DataParallel(student, device_ids=[0])
        #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    train_teacher(teacher, train_loader,test_loader,100)
    #load_model(net, 'models/mnist_gpu.pt')
    #load_model(net, 'models/mnist.pt')
    #load_model(net, 'models/cifar10_gpu.pt')
    #load_model(student,'./models/dd_cifar_student100.pt')
    #load_model(teacher,'./models/dd_cifar_teacher100.pt')
    #test_cifar10(teacher, test_loader)
    #load_model(student,'./models/dd_mnist_student100.pt')
    #train_cifar10(net,train_loader) 
    train_student(student, teacher, train_loader,test_loader,100)
    #test_cifar10(net, test_loader)
    #test_cifar10(student, test_loader)
    #save_model(net,'./models/mnist.pt')
    #net.eval()

