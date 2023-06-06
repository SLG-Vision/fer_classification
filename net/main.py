import torch
import torch.nn as nn
import torch.optim as optim
from VGG import VGG19
from CK import CK
import transforms as transforms
import os
import numpy as np

transform_train = transforms.Compose([
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.TenCrop(44),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

train = CK(split='Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, num_workers=1)
test = CK(split='Testing', transform=transform_test)
testloader = torch.utils.data.DataLoader(test, batch_size=5, shuffle=False, num_workers=1)

net = VGG19()

if torch.cuda.is_available():
    net.cuda()
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

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
    best_test_acc = 0
    
    net.eval()
    
    for batch_id, (inputs, labels) in enumerate(testloader):
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        
        if torch.cuda.is_available:
            inputs, labels = inputs.cuda(), labels.cuda()
        
        outputs = net(inputs)
        
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1) 
        loss = criterion(outputs_avg, labels)
        
        _, predicted = torch.max(outputs_avg.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        test_acc = 100.*correct/total
        
    if test_acc > best_test_acc:
        
        print(f"best_test_acc: {test_acc}")
        
        torch.save(net.state_dict(), os.path.join('.', 'model.t7'))
        best_test_acc = test_acc

for epoch in range(0, 60):
    train(epoch)
    test(epoch)