import torch
import os
from dataset import TEST, LABEL_MAP
from train import label2onehot, transform_labeled
from torch.utils.data import DataLoader
from train import BersonNetwork


data_path = 'Animals/Animals_Dataset'
device = 'cpu'
labels = []
test_path = os.path.normpath((os.path.join(data_path, 'test')))


test_dataset = TEST(
    test_path,
    transform = transform_labeled,
    target_transform = label2onehot 
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16
)
model = BersonNetwork()
model.load_state_dict(torch.load('model.path'))
model.eval()

with torch.no_grad():
  for _, images in enumerate(test_loader):
    y = model(images.to(device))
    print(y)
    batch_labels = torch.argmax(y, dim=1)
    labels.append(batch_labels)
ans = torch.cat(labels, 0).cpu().numpy()
print(ans)
print([LABEL_MAP[i] for i in ans])