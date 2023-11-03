import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class ECSSD(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.images = list(sorted(os.listdir(os.path.join(root_dir, 'images'))))
        self.masks = list(sorted(os.listdir(os.path.join(root_dir, 'ground_truth_mask'))))
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file = os.path.join(self.root_dir, 'images', self.images[idx])
        mask_file = os.path.join(self.root_dir, 'ground_truth_mask', self.masks[idx])
        
        image = Image.open(img_file)
        mask = Image.open(mask_file)
        if self.transforms:
            image = self.transforms(image)
        
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)

        return image, masks