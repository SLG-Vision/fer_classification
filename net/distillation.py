import torch
import torch.nn as nn
import torch.optim as optim

class Distillation(torch.Module):
    def __init__(self, teacher, student) -> None:
        super().__init__()
        self.teacher = teacher
        self.student = student      