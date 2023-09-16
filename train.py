import os
import torch
from torchvision import transforms
from torchvision.datasets import   ImageFolder
from dataset import TEST
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import time

NUM_CLASSES = os.environ['NUM_CLASSES']
LABELS = os.environ['LABELS']
LABEL_MAP = os.environ['LABEL_MAP']

label2onehot = transforms.Lambda(lambda y: torch.zeros(
    NUM_CLASSES, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

transform_labeled = transforms.Compose(
    [
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ]
)

class BersonNetwork(nn.Module):
  def __init__(self):
    super(BersonNetwork, self).__init__()
    self.flatten = nn.Flatten()
    self.c1 = nn.Conv2d(3, 20, 5, 2, 0)
    self.c2 = nn.Conv2d(20, 1, 5, 1, 0)
    self.linear_relu_stack = nn.Sequential(
        nn.Linear(676, 128), #!TODO: Change 0 to a proper value
        #!Tips: You should calculate the number of neurons.
        nn.ReLU(),
        nn.Linear(128, 512),
        nn.ReLU(),
        nn.Linear(512, NUM_CLASSES),#!TODO: Change 0 to a proper value
    )

  def forward(self, x):
    x = self.c1(x)
    x = F.relu(x) #!Question: What's the difference between torch.nn.relu() and torch.nn.F.relu()
    x = self.c2(x)
    x = F.relu(x)
    x = x.view(x.size(0), -1)
    logits = self.linear_relu_stack(x)
    return logits
  
class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self, name, fmt=':f'):
      self.name = name
      self.fmt = fmt
      self.reset()

  def reset(self):
      self.val = 0
      self.avg = 0
      self.sum = 0
      self.count = 0

  def update(self, val, n=1):
      self.val = val
      self.sum += val * n
      self.count += n
      self.avg = self.sum / self.count

  def __str__(self):
      fmtstr = '{name} {avg' + self.fmt + '}'
      return fmtstr.format(**self.__dict__)

def accuracy(output:torch.Tensor, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    print(pred.shape, target.shape)
    pred = pred.t()
    print(target.view(1, -1).shape)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def main():
    data_path = 'Animals/Animals_Dataset' #@param
    train_path = os.path.normpath((os.path.join(data_path, 'train')))

    batch_size = 16 #@param
    num_workers = 0 #@param
    learning_rate = 1e-3 #@param
    batch_size = 64 #@param
    epochs = 5 #@param
    
    
    train_dataset = ImageFolder(
        train_path,
        transform_labeled,
        target_transform = label2onehot
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model = BersonNetwork().to(device)

    start = time.time()
    for i in range(epochs):
        for batch, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)
            data_time.update(time.time() - start)
            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # acc1, acc5 = accuracy(pred, y, topk=(1, 5))
            losses.update(loss.item(), X.size(0))
            # top1.update(acc1[0], X.size(0))
            # top5.update(acc5[0], X.size(0))

        batch_time.update(time.time() - start)
        start = time.time()

        print(f"Epoch:{i + 1}: {batch_time}, {losses}")
    
    torch.save(model, 'model.path')

if __name__ == '__main__':
   main()