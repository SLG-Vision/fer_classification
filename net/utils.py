import torch
from torchvision import transforms
import cv2
import numpy as np


class Utility:
    def __init__(self):
        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAdjustSharpness(sharpness_factor=2),
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])

    @property
    def transform(self):
        return self._transform
    def print(self, frame):
        frame = frame.permute(1,2,0)
        img = frame.numpy()
        cv2.imshow('Transformed Img', img)

