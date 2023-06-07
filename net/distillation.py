from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Distillation:
    def __init__(self) -> None:
        self.distillation_loss = nn.KLDivLoss(reduction='batchmean')
        if torch.cuda.is_available():
            self.distillation_loss.cuda()
        self.temperature = 1
        self.alpha = 0.25

    def __call__(self, student_logits, student_target_loss, teacher_logits):
        
        distillation_loss = self.distillation_loss(F.log_softmax(student_logits / self.temperature, dim=1),
                                                   F.softmax(teacher_logits / self.temperature, dim=1))

        loss = (1 - self.alpha) * student_target_loss + self.alpha * distillation_loss
        return loss