import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms, datasets
from distillation import Distillation
from pretrained.VGG import VGG19
from FERNet import FERNet

label = {'angry':0, 'disgust': 1, 'fear':2, 'happy': 3, 'sad': 4, 'surprise':5, 'neutral':6}

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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True)

# Teacher: VGG19 pretrained
teacher = VGG19()
teacher.load_state_dict(torch.load('model.t7'))

# Student: FERNet
student = FERNet()

# Freeze teacher network
for layers in teacher.parameters():
    layers.requires_grad = False
    
# Distillator
distillator = Distillation()

# Optimizer
optimizer = optim.Adam(student.parameters(), lr=0.001)

#Student loss
criterion = nn.CrossEntropyLoss()

def train(epoch):
    for epoch in range(epoch):    
        for batch_id, (img, label) in enumerate(trainloader):
            
            teacher_outputs = teacher(img)
            student_outputs = student(img)

            loss_student = criterion(student_outputs, label)

            loss_dist = distillator(student_logits=student_outputs, teacher_logits=teacher_outputs, student_target_loss=loss_student)

            # Aggiorno i pesi dello student
            optimizer.zero_grad()
            loss_dist.backward()
            optimizer.step()
            
for epoch in range(60):
    train(epoch)