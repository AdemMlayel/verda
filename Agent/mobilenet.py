import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torchvision.models import mobilenet_v2
from torchsummary import summary
from torch.cuda.amp import GradScaler, autocast
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import time
import multiprocessing

# Enable cuDNN benchmark for faster fixed-size convolution operations
torch.backends.cudnn.benchmark = True
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.makedirs('checkpoints', exist_ok=True)

# === Device ===
def get_default_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)
    
class SmoothedCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        logprobs = F.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        return (self.confidence * nll_loss + self.smoothing * smooth_loss).mean()

# === Data ===
data_dir = r"C:\Users\GIGABYTE\Desktop\model\mobileNet\data"
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "val")

light_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

strong_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2))
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(train_dir, transform=light_train_transform)
valid_dataset = ImageFolder(valid_dir, transform=val_transform)

# === Class Weights ===
targets = train_dataset.targets
class_labels = np.unique(targets)
class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=targets)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
loss_fn = SmoothedCrossEntropy(smoothing=0.05)

# === DataLoader Splits ===
val_size = int(0.8 * len(valid_dataset))
test_size = len(valid_dataset) - val_size
val_dataset, test_dataset = random_split(valid_dataset, [val_size, test_size])

batch_size = 32
device = get_default_device()

train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_dl = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
test_dl = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
test_dl = DeviceDataLoader(test_dl, device)

# === Model ===
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        mixed_images, labels_a, labels_b, lam = mixup_data(images, labels)
        with autocast():
            out = self(mixed_images)
            loss = mixup_criterion(out, labels_a, labels_b, lam)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = loss_fn(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        batch_accuracy = [x['val_accuracy'] for x in outputs]
        return {
            'val_loss': torch.stack(batch_losses).mean(),
            'val_accuracy': torch.stack(batch_accuracy).mean()
        }

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], last_lr: {self.last_lr:.5f}, train_loss: {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}, val_acc: {result['val_accuracy']:.4f}")

class MobileNetV2Classifier(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        self.network = mobilenet_v2(weights=None)
        self.network.classifier[1] = nn.Linear(self.network.last_channel, num_classes)

    def forward(self, xb):
        return self.network(xb)

# === Mixup Helpers ===
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(pred, y_a, y_b, lam):
    return lam * loss_fn(pred, y_a) + (1 - lam) * loss_fn(pred, y_b)

# === Accuracy ===
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

# === Training Loop ===
def fit_OneCycle(epochs, max_lr, model, train_loader_func, val_loader, grad_clip=None, weight_decay=0, opt_func=torch.optim.AdamW):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader_func(0)))
    scaler = GradScaler()
    best_val_acc = 0.0

    for epoch in range(epochs):
        train_loader = train_loader_func('light' if epoch < epochs // 2 else 'strong')
        if epoch == epochs // 2:
            print("\U0001F501 Switching to full transform pipeline starting from epoch {}...".format(epoch))

        model.train()
        train_losses = []
        lrs = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()
            with autocast():
                loss = model.training_step(batch)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            lrs.append(optimizer.param_groups[0]['lr'])
            sched.step()
            train_losses.append(loss.detach().cpu())
            progress_bar.set_postfix(loss=loss.item())

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.last_lr = lrs[-1]
        model.epoch_end(epoch, result)
        history.append(result)

        ckpt_path = f"checkpoints/checkpoint_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

        if result['val_accuracy'] > best_val_acc:
            best_val_acc = result['val_accuracy']
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            print("✅ Best model updated: checkpoints/best_model.pth")

    return history

def make_train_loader(transform_type):
    if transform_type == 'light':
        transform = light_train_transform
    else:
        transform = strong_train_transform
    dataset = ImageFolder(train_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return DeviceDataLoader(loader, device)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    model = to_device(MobileNetV2Classifier(len(train_dataset.classes)), device)
    summary(model, (3, 224, 224))
    history = [evaluate(model, val_dl)]

    history += fit_OneCycle(
        epochs=50,
        max_lr=0.0045,
        model=model,
        train_loader_func=make_train_loader,
        val_loader=val_dl,
        grad_clip=0.1,
        weight_decay=1e-4
    )

    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    print("\nValidation Set Evaluation:")
    evaluate(model, val_dl)
    print("\nFinal Test Set Evaluation:")
    test_result = evaluate(model, test_dl)
    print(test_result)
