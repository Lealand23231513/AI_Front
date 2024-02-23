import torch
import time
import torch.nn as nn
import os
from torchvision.datasets import VisionDataset, ImageFolder
from PIL import Image
import torchvision.models as models
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader

from dataset import TEST, NUM_CLASSES

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

label2onehot = transforms.Lambda(lambda y: torch.zeros(
    NUM_CLASSES, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))



if __name__ == '__main__':
    
    '''
    微调预训练的ResNet18
    '''

    # hyperparameters
    learning_rate = 1e-3 #@param
    batch_size = 64 #@param
    epochs = 50 #@param
    sample_nums = 110
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    
    data_path = 'Animals/Animals_Dataset' #@param
    batch_size = 16 #@param
    num_workers = 8 #@param
    
    train_path = os.path.normpath((os.path.join(data_path, 'train')))
    
    train_dataset = ImageFolder(
        train_path,
        preprocess,
        target_transform = label2onehot
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle = True
    )
    
    
    model = models.resnet18(weights = torchvision.models.ResNet18_Weights.DEFAULT)

    # reolace the full connection layer
    fc_infeatures = model.fc.in_features
    model.fc = nn.Linear(fc_infeatures, NUM_CLASSES)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    fc_params_id = list(map(id, model.fc.parameters()))     
    base_params = filter(lambda p: id(p) not in fc_params_id, model.parameters())
    optimizer = torch.optim.SGD([
        {'params': base_params, 'lr': learning_rate*0},   # frozen conv layers
        {'params': model.fc.parameters(), 'lr': learning_rate}], momentum=0.9) # learn the params in full connection layer



    for i in range(epochs):
        start = time.time()
        losses = []
        for batch, (image, label) in enumerate(train_loader):
            image = image.to(device)
            label = label.to(device)
            pred = model(image)
            loss = loss_fn(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            batch_time = round(time.time()-start, 4)
        batch_loss = round(sum(losses)/len(losses), 4)
        print(f"Epoch:{i + 1}: batch_time = {batch_time}, average batch loss = {batch_loss}")

    torch.save(model, 'model.path')