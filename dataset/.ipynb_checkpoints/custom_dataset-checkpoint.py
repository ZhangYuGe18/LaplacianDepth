import os
import cv2
import tifffile as tiff
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from dataset.transform import Resize, NormalizeImage, PrepareForNet
from tqdm import tqdm
import random
class CustomDataset(Dataset):
    def __init__(self, image_paths, depth_paths, mode, size=(518, 518)):
        if mode != 'val' and mode != 'train':
            raise NotImplementedError
        
        self.mode = mode
        self.size = size
        self.image_paths = image_paths
        self.depth_paths = depth_paths
        
        net_w, net_h = size
        self.transform = Compose([
            Resize(
                width=net_w,
                height=net_h,
                resize_target=True if mode == 'train' else False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
    
    def __getitem__(self, item):
        img_path = self.image_paths[item]
        depth_path = self.depth_paths[item]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        depth = tiff.imread(depth_path).astype('float32')

        sample = self.transform({'image': image, 'depth': depth})
        sample['image'] = torch.from_numpy(sample['image'])
        sample['depth'] = torch.from_numpy(sample['depth'])#.unsqueeze(0)
        sample['depth'] = sample['depth'] / 256.0
        
        sample['valid_mask'] = sample['depth'] > 0
        
        sample['image_path'] = img_path
        
        return sample

    def __len__(self):
        return len(self.image_paths)

def get_image_path(root_path):
    image_paths = []
    depth_paths = []
    for folder_name in os.listdir(root_path):
        folder_path = os.path.join(root_path,folder_name)
        for image_name in os.listdir(folder_path):
            if 'color' in image_name:
                image_paths.append(os.path.join(folder_path,image_name))
            if 'depth' in image_name:
                depth_paths.append(os.path.join(folder_path,image_name))
        print(f"""{folder_name} includs image:{len(image_paths)} and image:{len(depth_paths)} """)
    combined = list(zip(image_paths, depth_paths))
    random.shuffle(combined)
    split_index = int(0.8 * len(combined))
    train_combined = combined[:split_index]
    val_combined = combined[split_index:]
    train_image_paths, train_depth_paths = list(zip(*train_combined))
    val_image_paths, val_depth_paths = list(zip(*val_combined))
    return train_image_paths, train_depth_paths,val_image_paths, val_depth_paths