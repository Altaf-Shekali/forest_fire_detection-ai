import torch
import torch.nn as nn
from torchvision import models
from torch.optim import Adam
from pathlib import Path
from dataset_utils import create_dataloaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_model(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

def eval_one_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.inference_mode():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total

def main():
    train_root = "data/raw/Train"
    test_root = "data/raw/Test"

    # Load data
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        train_root, test_root, batch_size=16
    )

    # Build model
    model = build_model(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    # Ensure models/ directory exists
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    best_model_path = models_dir / "best_model.pth"

    best_val = 0.0

    # Train loop
    for epoch in range(5):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion)

        print(f"Epoch {epoch+1}: Train Acc {tr_acc:.3f} | Val Acc {val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✔ Saved new best model with val_acc = {val_acc:.3f}")

if __name__ == "__main__":
    main()
