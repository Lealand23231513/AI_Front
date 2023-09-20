from PIL import Image
from torchvision.datasets import  VisionDataset
import os
NUM_CLASSES = 22
LABELS = [
    "ape", "bear", "bison", "cat",
    "chicken", "cow", "deer", "dog",
    "dolphin", "duck", "eagle", "fish",
    "horse", "lion", "lobster", "pig",
    "rabbit", "shark", "snake", "spider",
    "turkey", "wolf"
]
LABEL_MAP = {
    0: "ape", 1: "bear", 2: "bison", 3: "cat",
    4: "chicken", 5: "cow", 6: "deer", 7: "dog",
    8: "dolphin", 9: "duck", 10: "eagle", 11: "fish",
    12: "horse", 13: "lion", 14: "lobster",
    15: "pig", 16: "rabbit", 17: "shark", 18: "snake",
    19: "spider", 20:  "turkey", 21: "wolf"
}


class TEST(VisionDataset):
    def __init__(self,root, sample_nums, transform=None, target_transform=None):
        self.img_dir = root
        self.transform = transform
        self.target_transform = target_transform
        # self.samples = os.listdir(self.img_dir)
        self.samples = [f'{i}.png' for i in range(sample_nums)]

    def __len__(self):
        
        return len(self.samples)

    def __getitem__(self, index):
        img_path = os.path.normpath(os.path.join(self.img_dir, self.samples[index]))
        with open(img_path, "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img





