from pipeline.fer_classification.net.network import FERNet
from pipeline.fer_classification.net.pretrained.VGG import VGG19
from pipeline.fer_classification.net.utils import Utility
import torch

fernet_label = {0: 'angry', 1: 'disgust', 2: 'fear', 3:'sad' , 4: 'happy', 5: 'surprise', 6: 'neutral'}

class FER:

    def __init__(self, prod=True):
        self._net = FERNet() if prod else VGG19()
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._utils = Utility()

    def load(self, prod=True):
        self._net.load_state_dict(
            torch.load('fer_classification/net/student_distilled.t7' if prod else 'fer_classification/net/model.t7',
                       map_location=self._device))

    def predict(self, return_frame):
        input_tensor = self._utils.transform(return_frame)

        expanded_tensor = torch.zeros((3,48,48))
        for i in range(3):
            expanded_tensor[i,:,:] = input_tensor[0,:,:]
        self._utils.print(expanded_tensor)

        expanded_tensor = expanded_tensor.unsqueeze(0)
        pred = self._net(expanded_tensor)
        return fernet_label[int(torch.argmax(pred).item())]