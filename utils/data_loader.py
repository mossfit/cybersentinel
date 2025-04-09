import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class MalwareDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        
        for label_str in os.listdir(image_dir):
            label_path = os.path.join(image_dir, label_str)
            if os.path.isdir(label_path):
                for fname in os.listdir(label_path):
                    self.image_paths.append(os.path.join(label_path, fname))
                    self.labels.append(int(label_str))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

def get_data_loaders(train_dir, val_dir, batch_size=64, image_size=(160, 160)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = MalwareDataset(train_dir, transform=transform)
    val_dataset = MalwareDataset(val_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader
