import torch
import torch.nn as nn
import torch.optim as optim
from VGG import VGG19
import os
import numpy as np
from metrics import compute_metric
from db_loader import DBLoader

best_test_acc = 0
n_epoch = 75
db_name = 'FER2013'
metric = compute_metric(db_name)
db = DBLoader(db_name)

trainloader, testloader = db.load()

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
        torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
        
        # Calcolo dei gradienti
        loss.backward()
        
        # Aggiornamento dei parametri
        optimizer.step()
        
        loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        accuracy = 100.*correct / total
        #print(f"Accuracy {accuracy} Loss {loss/(batch_id+1)}")
        metric.add(outputs, accuracy, predicted, labels)
        
    metric.update()

    
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

        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        #print("Labels:")
        #print(labels)
        #print("Predicted:")
        #print(predicted)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        #print(f'predicted: {correct} on {total}')
        
        test_acc = 100.*correct/total
        metric.add_test(outputs, test_acc, predicted, labels)
        #print(f'Test_ACC:{test_acc}')
        
    if test_acc > best_test_acc:
        
        print(f"best_test_acc: {test_acc}")
        
        torch.save(net.state_dict(), os.path.join('.', f'vgg_{db_name}.t7'))
        best_test_acc = test_acc
        
    metric.update_test()
      
for epoch in range(0, n_epoch):
    train(epoch)
    test(epoch)
    
metric.plot(np.linspace(1, n_epoch, n_epoch).astype(int), metric='Loss')
metric.plot(np.linspace(1, n_epoch, n_epoch).astype(int), metric='Accuracy')
metric.plot(np.linspace(1, n_epoch, n_epoch).astype(int), metric='F1-Score')
metric.plot(np.linspace(1, n_epoch, n_epoch).astype(int), metric='Precision')
metric.plot(np.linspace(1, n_epoch, n_epoch).astype(int), metric='Recall')
