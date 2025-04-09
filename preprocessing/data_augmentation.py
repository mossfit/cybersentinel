from torchvision import transforms

def get_augmentation_transform(image_size=(160, 160)):
     augmentation = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])
    return augmentation
