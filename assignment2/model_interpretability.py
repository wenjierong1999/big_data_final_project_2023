import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import random
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
from pytorch_grad_cam import GradCAM
import matplotlib.pyplot as plt
import torchcam
from torchcam.utils import overlay_mask

test_transform = transforms.Compose([transforms.Resize(512),
                                     # transforms.CenterCrop(512),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])
checkpoint_path = '/tmp/pycharm_project_444/assignment2/model_checkpoint_q2/a2/'
model_path = os.path.join(checkpoint_path,os.listdir(checkpoint_path)[0])
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = torch.load(model_path).eval().to(device)

exp_food_img_path = '/root/autodl-tmp/images/1075320.jpg'
exp_interior_img_path ='/root/autodl-tmp/images/1182868.jpg'


if __name__ == '__main__':
    exp_img = Image.open(exp_food_img_path)
    input_tensor = test_transform(exp_img).unsqueeze(0).to(device)

    target_layer = model.layer4[-1] #Target layer we intend to analyze
    print(target_layer)
    targets = [ClassifierOutputTarget(0)]

    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=True)
    cam_map = cam(input_tensor=input_tensor, targets=targets)[0]
    plt.imshow(cam_map)
    plt.show()

    result = overlay_mask(exp_img, Image.fromarray(cam_map), alpha=0.6)  # alpha越小，原图越淡
    result.save('/tmp/pycharm_project_444/assignment2/cam.jpg')