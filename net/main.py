import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from distillation import Distillation
from pretrained.VGG import VGG19
from FERNet import FERNet
from torch.nn.parallel import DataParallel

label = {'angry':0, 'disgust': 1, 'fear':2, 'happy': 3, 'sad': 4, 'surprise':5, 'neutral':6}
best_test_acc = 0

# Definizione delle trasformazioni per le immagini
transform = transforms.Compose([
    transforms.Resize((48, 48)),  # Ridimensiona le immagini a una dimensione specifica
    transforms.ToTensor(),  # Converte l'immagine in un tensore
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizza i valori dei pixel dell'immagine
])

# Caricamento del dataset di immagini
trainset = datasets.ImageFolder(root=f'./fer2013/train/', transform=transform)
testset = datasets.ImageFolder(root=f'./fer2013/test/', transform=transform)

# Creazione del dataloader per caricare i dati in batch
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)

device_0 = torch.device("cuda:0")
device_1 = torch.device("cuda:1")

# Teacher: VGG19 pretrained
teacher = VGG19()
teacher.load_state_dict(torch.load('model.t7'))

# Freeze teacher network
for layers in teacher.parameters():
    layers.requires_grad = False
    
teacher = teacher.to(device_0)
teacher = DataParallel(teacher, device_ids=[device_0])

# Student: FERNet
student = FERNet()
student = student.to(device_1)
student = DataParallel(student, device_ids=[device_1])

# Distillator
distillator = Distillation()

# Optimizer
optimizer = optim.SGD(student.parameters(), lr=0.001)

#Student loss
criterion = nn.CrossEntropyLoss().to(device_1)
criterion = DataParallel(criterion, device_ids=[device_1])

def train(epoch):
    print(f'Training epoch: {epoch}')
    student.train()
    total = 0
    correct = 0
    for epoch in range(epoch):    
        for batch_id, (img, label) in enumerate(trainloader):
            
            img_0 = img.to(device_0) 
            img_1 = img.to(device_1)
            label = label.to(device_1)

            teacher_outputs = teacher(img_0)
            teacher_outputs = teacher_outputs.to(device_1)
            
            student_outputs = student(img_1)
            student_outputs = student_outputs.to(device_1)
            
            loss_student = criterion(student_outputs, label)

            loss_dist = distillator(student_logits=student_outputs, teacher_logits=teacher_outputs, student_target_loss=loss_student)

            # Aggiorno i pesi dello student
            optimizer.zero_grad()
            loss_dist.backward()
            optimizer.step()
            
            loss_dist += loss_dist.data.item()
            _, predicted = torch.max(student_outputs.data, 1)            
            total += label.size(0)
            
            correct += (predicted.to(device_1) == label).sum().item()

            accuracy = correct / total
        print(f"Accuracy {accuracy} Distillation Loss {loss_dist/(batch_id+1)}")


def test(epoch):
    total = 0
    correct = 0
    test_acc = 0
    global best_test_acc
    print(f'Testing epoch {epoch}')
    student.eval()
    
    for batch_id, (inputs, labels) in enumerate(testloader):
        
        inputs, labels = inputs.to(device_1), labels.to(device_1)
        
        student_outputs = student(inputs)
        student_outputs = student_outputs.to(device_1)
        loss = criterion(student_outputs, labels)
        
        _, predicted = torch.max(student_outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.to(device_1) == labels).sum().item()
        
        test_acc = 100.*correct/total
        
    if test_acc > best_test_acc:
        
        print(f"best_test_acc: {test_acc}")
        
        torch.save(student.state_dict(), os.path.join('.', 'student_distilled.t7'))
        best_test_acc = test_acc        

for epoch in range(0, 30):
    train(epoch)
    test(epoch)