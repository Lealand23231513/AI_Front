import torch
import os
from dataset import TEST, LABEL_MAP
from train import preprocess
from torch.utils.data import DataLoader

data_path = 'Animals/Animals_Dataset'
device = "cuda" if torch.cuda.is_available() else "cpu"
labels = []
test_path = os.path.normpath((os.path.join(data_path, 'test')))
sample_nums = 110


test_dataset = TEST(
    test_path,
    sample_nums,
    transform = preprocess
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16
)
if __name__ == '__main__':
    model = torch.load('model.path', map_location=torch.device(device))
    with torch.no_grad():
      for _, images in enumerate(test_loader):
        y = model(images.to(device))
        batch_labels = torch.argmax(y, dim=1)
        labels.append(batch_labels)
    ans = torch.cat(labels, 0).cpu().numpy()
    print([f'{id}.png : {LABEL_MAP[i]}' for id,i in enumerate(ans)])