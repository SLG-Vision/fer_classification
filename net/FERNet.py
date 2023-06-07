import torch
import torch.nn as nn

class FERNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc = nn.Linear(4 * 112 * 112, 2)

    def forward(self, x):
        out = self.layer1(x)
        
        out = out.view(out.size(0), -1)
        
        out = self.fc(out)
        return out