import os
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_transforms(image_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    return train_transform, eval_transform

def create_dataloaders(train_root, test_root, batch_size=32, val_split=0.2, image_size=224):
    train_transform, eval_transform = get_transforms(image_size)
    full_train_dataset = datasets.ImageFolder(root=train_root, transform=train_transform)

    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    val_dataset.dataset.transform = eval_transform
    test_dataset = datasets.ImageFolder(root=test_root, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names = full_train_dataset.classes
    return train_loader, val_loader, test_loader, class_names
