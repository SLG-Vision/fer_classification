import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torchvision import transforms,datasets
from distillation import Distillation
from pretrained.VGG import VGG19
from FERNet import FERNet
from pretrained.CK import CK


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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False, num_workers=1)

# Teacher: VGG19 pretrained
teacher = VGG19()
teacher.load_state_dict(torch.load('vgg19_newdb.t7'))
teacher = teacher.cuda()

# Student: FERNet
student = FERNet()
student = student.cuda()

# Distillator
distillator = Distillation()

# Optimizer
optimizer = optim.SGD(student.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

#Student loss
criterion = nn.CrossEntropyLoss().cuda()

def train(epoch):
    print(f'Training epoch: {epoch}')
    student.train()
    total = 0
    correct = 0
    teacher_correct = 0
    teacher_acc = 0
    for batch_id, (img, label) in enumerate(trainloader):
        
        img = img.cuda() 
        label = label.cuda()
        
        optimizer.zero_grad()

        with torch.no_grad():
            teacher_outputs = teacher(img)
        
        teacher_outputs = teacher_outputs.cuda()
        
        student_outputs = student(img)
        student_outputs = student_outputs.cuda()

        
        loss_student = criterion(student_outputs, label)
        loss_dist = distillator(student_logits=student_outputs, teacher_logits=teacher_outputs, student_target_loss=loss_student)
        
        # Aggiorno i pesi dello student
        loss_dist.backward()
        
        for group in optimizer.param_groups:
            for param in group['params']:
                param.grad.data.clamp_(-0.1, 0.1)
                
        optimizer.step()
        
        loss_dist += loss_dist.data.item()
        _, predicted = torch.max(student_outputs.data, 1)
        _, teacher_predicted = torch.max(teacher_outputs.data, 1)            
            
        total += label.size(0)
        
        correct += (predicted.cuda() == label).sum().item()
        teacher_correct += (teacher_predicted.cuda() == label).sum().item()
        accuracy = 100.* correct / total
        teacher_acc = 100.*teacher_correct/total
        
        print(f"\tStudent_Accuracy {accuracy}\tTeacher_Accuracy {teacher_acc} \tDistillation Loss {loss_dist/(batch_id+1)}")  
        
        
def test(epoch):
    
    total = 0
    correct = 0
    test_acc = 0
    counter = 0
    best_loss = 0
    patient = 10
    student.eval()
    global best_test_acc
    print(f'Testing epoch {epoch}')    
    
    for batch_id, (inputs, labels) in enumerate(testloader):
        
        inputs, labels = inputs.cuda(), labels.cuda()
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        
        student_outputs = student(inputs)
        student_outputs = student_outputs.cuda()

        student_outputs = student_outputs.view(bs, ncrops, -1).mean(1)
        loss = criterion(student_outputs, labels)
        
        if loss > best_loss:
            best_loss = loss
            counter = 0
        else:
            counter += 1
        
        _, predicted = torch.max(student_outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cuda() == labels).sum().item()
        
        test_acc = 100.*correct/total
        print(f"\tAccuracy {test_acc}\t Correct/Total {correct}/{total}")  
        
        if counter > patient:
            print(f'Early Stopping at {epoch}')
            break

    if test_acc > best_test_acc:
        print(f"best_test_acc: {test_acc}")
        student_save = student.to("cpu")
        torch.save(student_save.state_dict(), os.path.join('.', 'fernet_res_dist.t7'))
        student_save = student.cuda()
        best_test_acc = test_acc        


for epoch in range(0, 60):
    train(epoch)
    test(epoch)