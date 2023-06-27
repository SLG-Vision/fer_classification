import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets
from distillation import Distillation
from pretrained.VGG import VGG19
from network import FERNet

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

print(len(trainset), len(testset))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(testset, batch_size=20, shuffle=False, num_workers=1)

print(len(trainloader), len(testloader))

# Teacher: VGG19 pretrained
teacher = VGG19()
teacher.load_state_dict(torch.load('./pretrained/vgg_best.t7'))

# Freeze teacher network
for layers in teacher.parameters():
    layers.requires_grad = False
    
# Student: FERNet
student = FERNet()

# Distillator
distillator = Distillation()

# Optimizer
optimizer = optim.SGD(student.parameters(), lr=0.001)

#Student loss
criterion = nn.CrossEntropyLoss()

def train(epoch):
    print(f'Training epoch: {epoch}')
    student.train()
    total = 0
    correct = 0
    
    for batch_id, (img, label) in enumerate(trainloader):        
        
        img, label = img.cuda(), label.cuda()
        
        teacher_outputs = teacher(img)
        student_outputs = student(img)
        
        loss_student = criterion(student_outputs, label)
        loss_dist = distillator(student_logits=student_outputs, teacher_logits=teacher_outputs, student_target_loss=loss_student)
        
        # Aggiorno i pesi dello student
        optimizer.zero_grad()
        
        loss_dist.backward()
        
        optimizer.step()
        
        loss_dist += loss_dist.data.item()
        _, predicted = torch.max(student_outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        accuracy = correct / total
        
        print(f"Accuracy {accuracy} Distillation Loss {loss_dist/(batch_id+1)}")

def test(epoch):
    total = 0
    correct = 0
    test_acc = 0
    global best_test_acc
    print(f'Testing epoch {epoch}')
    student.eval()
    
    for (inputs, labels) in testloader:
        
        inputs, labels = inputs.cuda(), labels.cuda()
        
        student_outputs = student(inputs)
        loss = criterion(student_outputs, labels)
        
        _, predicted = torch.max(student_outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        test_acc = 100.*correct/total
        print(f'predicted: {correct} on {total}')

    if test_acc > best_test_acc:
        
        print(f"best_test_acc: {test_acc}")
        
        torch.save(student.state_dict(), os.path.join('.', f'student_distilled_{test_acc}.t7'))
        best_test_acc = test_acc        

for epoch in range(0, 60):
    train(epoch)
    test(epoch)