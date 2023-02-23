import os
import torch
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from pathlib import Path
from io import BytesIO

# parameters
input_size = 224 
label_to_class_name = {0: 'Bed', 1: 'Chair', 2: 'Sofa'}
class_name_to_label = {value:key for key, value in label_to_class_name.items()}
num_classes = len(label_to_class_name.keys())
BASE_DIR = Path(__file__).resolve(strict=True).parent
saved_model_path = 'trained_model'

# load model
model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("{}/{}".format(BASE_DIR, saved_model_path), map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict_furniture(image_file):
    image = Image.open(BytesIO(image_file))
    image = image.convert("RGB")
    image = transform(image)
        
    image = torch.unsqueeze(image, dim=0)
        
    output = model(image.to(device))
    _, pred = torch.max(output, 1)
    predicted_label = pred[0].item()
    return label_to_class_name[predicted_label]