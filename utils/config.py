import torch

config = {
    'learning_rate': 1e-4,
    'batch_size': 64,
    'num_epochs': 30,
    'train_image_size': (160, 160),
    'aux_image_size': (20, 20),
    'num_classes': 25,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42
}
