import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from distillation import Distillation
from pretrained.VGG import VGG19
from network import FERNet
from pretrained.metrics import compute_metric
from pretrained.db_loader import DBLoader

best_test_acc = 0
n_epoch = 35
db_name = 'CK+'
metric = compute_metric(db_name)
db = DBLoader(db_name)

trainloader, testloader = db.load()

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
        accuracy = 100.*correct / total
        
        print(f"Accuracy {accuracy} Distillation Loss {loss_dist/(batch_id+1)}")
        metric.add(student_outputs, accuracy, predicted, label)
        
    metric.update()
    
def test(epoch):
    total = 0
    correct = 0
    test_acc = 0
    global best_test_acc
    print(f'Testing epoch {epoch}')
    student.eval()
    
    for (inputs, labels) in testloader:
                
        student_outputs = student(inputs)
        loss = criterion(student_outputs, labels)
        
        _, predicted = torch.max(student_outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        test_acc = 100.*correct/total
        metric.add_test(student_outputs, test_acc, predicted, labels)
        print(f'predicted: {correct} on {total}')

    if test_acc > best_test_acc:
        
        print(f"best_test_acc: {test_acc}")
        
        torch.save(student.state_dict(), os.path.join('.', f'student_distilled_ck_{test_acc}.t7'))
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
