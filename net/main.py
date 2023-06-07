import torch
import torch.optim as optim
from distillation import Distillation
from pretrained.VGG import VGG19
from FERNet import FERNet

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

def train():
    # Addestramento del modello studente con distillazione
    for epoch in range(60):
        for inputs, targets in dataloader:

            # Passaggio dei dati attraverso il modello insegnante per ottenere le logits
            teacher_logits = teacher(inputs)

            # Ottieni le logits del modello studente
            student_logits = student(inputs)

            # Calcola la perdita utilizzando la distillazione
            loss = distillator(F.log_softmax(student_logits, dim=1), F.softmax(teacher_logits, dim=1))

            # Aggiorna i pesi del modello studente
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Valutazione delle prestazioni del modello studente
        # ...

def test():
    pass