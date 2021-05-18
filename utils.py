from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import base64
import cv2


class Base64_Encoder(object):
    """Image pre-processing.

    Encode an image into base64
    """
    def __call__(self, image):
        image = np.float32(image)
        _, image = cv2.imencode('.jpg', image)
        return base64.b64encode(image)


class Base64_Decoder(object):
    """Image pre-processing.

    Decode base64 into image
    """
    def __call__(self, image):
        nparr = np.frombuffer(base64.b64decode(image), np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


TRAIN_DATA_TRANSFORM = transforms.Compose([
    Base64_Encoder(),
    Base64_Decoder(),
    transforms.ToPILImage(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.500, 0.500, 0.500],
                         std=[0.500, 0.500, 0.500])
])

VAL_DATA_TRANSFORM = transforms.Compose([
    Base64_Encoder(),
    Base64_Decoder(),
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.500, 0.500, 0.500],
                         std=[0.500, 0.500, 0.500])
])


def load_model(model_path, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.jit.load(model_path, map_location=device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def single_predict(model, image):
    cuda_available = torch.cuda.is_available()

    img_tensor = VAL_DATA_TRANSFORM(image).float()
    img_tensor = img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.cuda() if cuda_available else img_tensor.cpu()
    input_img = Variable(img_tensor)
    output = model(input_img)

    sm = nn.Softmax(dim=1)
    sm = sm.cuda() if cuda_available else sm.cpu()
    probabilities = sm(output)
    _, tensor = torch.max(output.data, 1)
    pred = tensor[0].item()

    trust_1 = probabilities[0][0].item()
    trust_2 = probabilities[0][1].item()

    if trust_1 > trust_2:
        trust = trust_1
    else:
        trust = trust_2

    return pred, trust


def save_models_metrics(model_name, metrics_filepath,
                        y_true, y_pred, t_pred=None):
    report = classification_report(y_true, y_pred,
                                   digits=4, target_names=['real', 'spoof'])
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=[0, 1]),
                      index=['true:real', 'true:spoof'],
                      columns=['pred:real', 'pred:spoof'])

    print(f' >> Relatorio:\n{report}\n\n')
    print(f' >> Matriz de Confusão: {cm}\n\n')

    if t_pred is not None:
        print(f' >> Pred medio: {t_pred.mean():.3f} +/- {t_pred.std():.3f} ms')

    with open(metrics_filepath, 'w') as f:
        f.write(' ' + '-'*50 + '\n')
        f.write(f'     MODELO: {model_name}\n')
        f.write(' ' + '-'*50 + '\n')

        f.write(' >> Relatório:\n\n')
        f.write(report)
        f.write('\n\n >> Matriz de Confusão:\n\n')
        f.write(f'{cm}\n\n')
        if t_pred is not None:
            f.write(f' >> Predict medio: {t_pred.mean():.1f} +/- ' +
                    f'{t_pred.std():.1f} ms\n')


def load_densenet(path=None):
    from torchvision.models.densenet import DenseNet
    import re

    model = DenseNet(32, (6, 12, 24, 16), 64, memory_efficient=True)
    pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    if path is not None:
        state_dict = torch.load(path)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model
