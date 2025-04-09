import torch
from utils.data_loader import get_data_loaders
from networks.cybersentinel_model import CyberSentinel
from training.trainer import train_model
from utils.config import config

def main():
    train_dir = './dataset/images/train'
    val_dir = './dataset/images/val'
    
    train_loader, val_loader = get_data_loaders(train_dir, val_dir, 
                                                batch_size=config['batch_size'], 
                                                image_size=config['train_image_size'])
    
    # Instantiate the model
    model = CyberSentinel(num_classes=config['num_classes'])
    
    # Train the model
    model = train_model(model, train_loader, val_loader, num_epochs=config['num_epochs'])
    
   
    
if __name__ == "__main__":
    main()
