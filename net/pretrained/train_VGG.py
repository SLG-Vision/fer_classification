import torch
import torch.nn as nn
import torch.optim as optim
from VGG import VGG19
from CK import CK
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np

best_test_acc = 0

transform_train = transforms.Compose([
    transforms.Grayscale(1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.Grayscale(1),
    transforms.ToTensor()
])

db = datasets.DatasetFolder(root='/workspace/FER Classification/Facial-Expression-Recognition.Pytorch/CK+48', transform=transform_train, loader=datasets.folder.default_loader, extensions='.png')
trainset_len = int(0.8*len(db))
trainset, testset = torch.utils.data.random_split(db, lengths=[trainset_len, len(db)-trainset_len])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(testset, batch_size=20, shuffle=False, num_workers=1)

net = VGG19()

if torch.cuda.is_available():
    net.cuda()
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

best_test_acc = 0

def train(epoch):
    print("Epoca:" + str(epoch))
    net.train()
    correct = 0
    total = 0
    
    for batch_id, (inputs, labels) in enumerate(trainloader):
        
        inputs, labels = inputs.cuda(), labels.cuda()
        
        # Azzeramento dei gradienti
        optimizer.zero_grad()
        
        # Forward pass
        outputs = net(inputs)
        
        # Calcolo della loss
        loss = criterion(outputs, labels)
        
        # Clip Gradient
        #torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
        
        # Calcolo dei gradienti
        loss.backward()
        
        # Aggiornamento dei parametri
        optimizer.step()
        
        loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f"Accuracy {accuracy} Loss {loss/(batch_id+1)}")
        
def test(epoch):
    total = 0
    correct = 0
    test_acc = 0
    global best_test_acc
    
    net.eval()
    
    for (inputs, labels) in testloader:
        bs, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        
        if torch.cuda.is_available:
            inputs, labels = inputs.cuda(), labels.cuda()
        
        #print('inputs: ')
        #print(inputs.shape)
        
        outputs = net(inputs)
        #print(inputs[0])
        #print(outputs[0])

        #outputs_avg = outputs.view(bs, -1).mean(1) 
        loss = criterion(outputs, labels)
        
        _, predicted = torch.max(outputs.data, 1)
        #print("Labels:")
        #print(labels)
        #print("Predicted:")
        #print(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(f'predicted: {correct} on {total}')
        
        test_acc = 100.*correct/total
        #print(f'Test_ACC:{test_acc}')
        
    if test_acc > best_test_acc:
        
        print(f"best_test_acc: {test_acc}")
        
        torch.save(net.state_dict(), os.path.join('.', f'vgg_{test_acc}.t7'))
        best_test_acc = test_acc
    
for epoch in range(0, 80):
    train(epoch)
    test(epoch)