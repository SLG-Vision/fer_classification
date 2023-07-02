import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torchvision import transforms,datasets
from pretrained.VGG import VGG19


best_test_acc = 0
transform_train = transforms.Compose([
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.TenCrop(44),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

# Creazione del dataloader per caricare i dati in batch
""""trainset = CK(split='Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)
testset = CK(split='Testing', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False, num_workers=1)"""

trainset=datasets.ImageFolder(root='./bigfer/dataset/images/train',transform=transform_train)
testset= datasets.ImageFolder(root='./bigfer/dataset/images/validation' ,transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False, num_workers=1)

# Teacher: VGG19 pretrained
net = VGG19()
net = net.cuda()

# Optimizer
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

#Student loss
criterion = nn.CrossEntropyLoss().cuda()

def train(epoch):
    print(f'Training epoch: {epoch}')
    net.train()
    total = 0
    correct = 0
    for batch_id, (img, label) in enumerate(trainloader):
        
        img = img.cuda() 
        label = label.cuda()
        
        optimizer.zero_grad()

        outputs = net(img)
        
        outputs = outputs.cuda()

        loss = criterion(outputs, label)
        
        # Aggiorno i pesi dello student
        loss.backward()
        
        for group in optimizer.param_groups:
            for param in group['params']:
                param.grad.data.clamp_(-0.1, 0.1)
                
        optimizer.step()
        
        loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)            
        total += label.size(0)
        
        correct += (predicted.cuda() == label).sum().item()
        accuracy = 100.* correct / total
        
        print(f"\tAccuracy {accuracy} \tDistillation Loss {loss/(batch_id+1)}")  
        
        
def test(epoch):
    
    total = 0
    correct = 0
    test_acc = 0
    counter = 0
    best_loss = 0
    patient = 10
    net.eval()
    global best_test_acc
    print(f'Testing epoch {epoch}')    
    
    for batch_id, (inputs, labels) in enumerate(testloader):
        
        inputs, labels = inputs.cuda(), labels.cuda()
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        outputs = net(inputs)

        outputs = outputs.view(bs, ncrops, -1).mean(1)
        loss = criterion(outputs, labels)
        
        if loss > best_loss:
            best_loss = loss
            counter = 0
        else:
            counter += 1
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cuda() == labels).sum().item()
        
        test_acc = 100.*correct/total
        print(f"\tAccuracy {test_acc}\t Correct/Total {correct}/{total}")  
        
        if counter > patient:
            print(f'Early Stopping at {epoch}')
            break

    if test_acc > best_test_acc:
        print(f"best_test_acc: {test_acc}")
        student_save = net.to("cpu")
        torch.save(student_save.state_dict(), os.path.join('.', 'vgg19_newdb.t7'))
        student_save = net.cuda()
        best_test_acc = test_acc        


for epoch in range(0, 60):
    train(epoch)
    test(epoch)