"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.mypath import MyPath

class DrewFish(Dataset):
    def __init__(self, main_dir=MyPath.db_root_dir('drew-fish'), transform=None):
        super(DrewFish, self).__init__()
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = [im for im in os.listdir(main_dir) if im.endswith('.jpg')]
        self.total_imgs = sorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, index):
        img_loc = os.path.join(self.main_dir, self.total_imgs[index])
        image = Image.open(img_loc).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        out = {'image': image}
        return out

    def get_image(self, idx):
        return Image.open(os.path.join(self.main_dir, self.total_imgs[idx])).convert("RGB")
