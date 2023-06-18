import torch
from torchvision import transforms
import cv2


class Utility:
    def __init__(self):
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((48, 48)),
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])

    @property
    def transform(self):
        return self._transform

    def print(self, frame):
        img = frame.numpy()
        img = img.transpose(1,2,0)
        cv2.imshow('Transformed Img', img)

