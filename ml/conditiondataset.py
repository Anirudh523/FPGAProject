import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class RoadConditionDataset(Dataset):
    def __init__(self, manifest_path:str, image_size:int = 128, augment: bool = False):
        self._pairs = []
        p = Path(manifest_path)
        
        with open(p, 'r',  encoding="utf-8") as fr:
            lines = fr.readlines()
            for line in lines[1:]:
                parts = line.strip().split(",")
                self._pairs.append((parts[0], parts[1]))
        
        if augment:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self._pairs)
    
    def __getitem__(self, idx):
        image_path, label_path = self._pairs[idx]

        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)

        label = np.load(label_path)
        grid_labels = label.astype(np.int64) + 1
        label_tensor = torch.from_numpy(grid_labels)

        return image_tensor, label_tensor
    
if __name__ == "__main__":
    dataset = RoadConditionDataset(
        "/Users/anirudh/RoadConditionClassifier/FPGAProject/ml/preprocessed/manifest.csv"
    )

    print(f"Dataset size: {len(dataset)}")

    image_tensor, label_tensor = dataset[0]
    print(f"Image tensor size: {image_tensor.shape}")
    print(f"Label tensor size: {label_tensor.shape}")
    print(f"Label tesnor values: {label_tensor}")


        


            
        
                
    
