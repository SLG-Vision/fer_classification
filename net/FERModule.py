from .network import FERNet
from .pretrained.VGG import VGG19
from .utils import Utility
import torch
import cv2

fernet_label = {0: 'anger', 1: 'contempt', 2: 'disgust', 3:'fear' , 4: 'happy', 5: 'sadness', 6: 'surprise'}

class FER:

    def __init__(self, prod=True):
        self._net = FERNet() if prod else VGG19()
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._utils = Utility()

    def load(self, prod=True):
        self._net.load_state_dict(
            torch.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'student_distilled.t7') if prod else 'fer_classification/net/model.t7',
                       map_location=self._device))

    def predict(self, return_frame):
        resized_image = cv2.resize(return_frame, (48, 48))
        input_tensor = self._utils.transform(resized_image)

        self._utils.print(input_tensor)

        input_tensor = input_tensor.unsqueeze(0)

        pred = self._net(input_tensor)

        return fernet_label[int(torch.argmax(pred).item())]