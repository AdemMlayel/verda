import os
import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from mobilenet import MobileNetV2Classifier

torch.backends.cudnn.benchmark = True

# === Paths ===
train_dir = r"C:\Users\GIGABYTE\Desktop\model\mobileNet\data\train"
val_dir = r"C:\Users\GIGABYTE\Desktop\model\mobileNet\data\val"
test_dir = r"C:\Users\GIGABYTE\Desktop\model\mobileNet\data\test"

# === Strong Data Augmentation ===
strong_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    transforms.RandomPerspective(distortion_scale=0.3, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Class Names ===
class_names = sorted(os.listdir(train_dir))

# === DataLoaders ===
val_dataset = ImageFolder(val_dir, transform=val_transform)
val_dl = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = ImageFolder(test_dir, transform=val_transform)
test_dl = DataLoader(test_dataset, batch_size=32, shuffle=False)

# === DeviceDataLoader ===
class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl:
            yield [x.to(self.device) for x in b]
    def __len__(self):
        return len(self.dl)

val_dl = DeviceDataLoader(val_dl, device)
test_dl = DeviceDataLoader(test_dl, device)

# === Evaluation Function ===
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    total, correct = 0, 0
    for images, labels in val_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    accuracy = correct / total
    print(f"✅ Evaluation Accuracy: {accuracy:.4f}")
    return accuracy

# === Fit Loop ===
def fit_OneCycle(epochs, max_lr, model, train_loader_func, val_loader, grad_clip=None, weight_decay=0, opt_func=torch.optim.AdamW):
    print("\n📊 Starting fine-tuning with val accuracy tracking and progress bar...")
    torch.cuda.empty_cache()
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader_func(0)))
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        train_loader = train_loader_func('strong')
        train_losses = []
        lrs = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True, dynamic_ncols=True)
        for batch in progress_bar:
            images, labels = batch
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            sched.step()
            train_losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])
            progress_bar.set_postfix({'loss': loss.item()})

        val_acc = evaluate(model, val_loader)
        print(f"Epoch [{epoch+1}], train_loss: {np.mean(train_losses):.4f}, last_lr: {lrs[-1]:.5f}, val_acc: {val_acc:.4f}")
    return model

# === Fine-Tuning Using Only Strong Transforms ===
def fine_tune_mobilenet_strong_aug(model_path, epochs=10, max_lr=0.0005):
    print("\n🚀 Fine-tuning MobileNet model with strong augmentations only...")
    model = MobileNetV2Classifier(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    def strong_loader(_):
        dataset = ImageFolder(train_dir, transform=strong_train_transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
        return DeviceDataLoader(loader, device)

    model = fit_OneCycle(
        epochs=epochs,
        max_lr=max_lr,
        model=model,
        train_loader_func=strong_loader,
        val_loader=val_dl,
        grad_clip=0.1,
        weight_decay=1e-4
    )
    print("\n✅ Fine-tuning complete. Evaluating on validation and test sets...")
    evaluate(model, val_dl)
    evaluate(model, test_dl)
    torch.save(model.state_dict(), "checkpoints/fine_tuned_mobilenet.pth")
    print("📦 Fine-tuned model saved as 'fine_tuned_mobilenet.pth'")

# === Run Fine-Tuning ===
if __name__ == '__main__':
    fine_tune_mobilenet_strong_aug("checkpoints/best_model.pth", epochs=10, max_lr=0.0005)
