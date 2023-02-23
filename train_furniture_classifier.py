import torch
import os
import glob
import torchvision
import time
import copy
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image


# DISCLAIMER: some code and parameters are adapted from official pytorch tutorials (https://pytorch.org/tutorials)

input_size = 224 

# load image data
data_dir = 'data'
class_names = [folder for folder in os.listdir(data_dir) if not folder.startswith('.')] # ignore hidden files
num_classes = len(class_names)
 
label_to_class_name = {i:name for i, name in enumerate(class_names)}
class_name_to_label = {value:key for key, value in label_to_class_name.items()}

images = []
for i in range(num_classes):
    images.append(glob.glob(data_dir+'/{}/*.jpg'.format(label_to_class_name[i]),recursive=True))

# divide data from each class into training and validation
#   (obtain an equal representation of each class in the training and validation dataset)
train_images = []
val_images = []
for i in range(num_classes):
    train_idx, val_idx = random_split(images[i], [0.7, 0.3], generator=torch.Generator().manual_seed(8))
    train_images.extend([images[i][j] for j in train_idx.indices])
    val_images.extend([images[i][j] for j in val_idx.indices])


# custom dataset
class FurnitureDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_path = self.img_dir[idx] 
        image = None
        with open(img_path, "rb") as f:
            image = Image.open(f)
            image = image.convert("RGB")
        label = class_name_to_label[img_path.split('/')[-2]]
        if self.transform:
            image = self.transform(image)
        return image, label
    
# create transformations (augment training data)
transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
    
# create datasets and dataloaders
train_dataset = FurnitureDataset(train_images, transform['train'])
val_dataset = FurnitureDataset(val_images, transform['val'])
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)


# training code adapted from https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html 
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# initialize training model: use Resnet18 as our starting point, finetune it to our dataset
model_ft = models.resnet18(weights='ResNet18_Weights.DEFAULT')
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)


# setting up training parameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)
params_to_update = model_ft.parameters()
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
dataloaders_dict = {'train': train_dataloader, 'val': val_dataloader}
num_epochs = 20


# train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)


# save the model
output_path = 'trained_model'
torch.save(model_ft.state_dict(), output_path)

