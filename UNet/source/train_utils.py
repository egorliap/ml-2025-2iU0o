import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import numpy as np

from model import UNet  
from plus_model import UNetPlus
from env import settings


def mask_to_long(mask, image_width: int, image_height: int):
    """
    Преобразует маску в LongTensor с числами 0,1,2
    В датасете 1 = животное, 2 = фон1 = фон, 2 = животное, 3 = граница
    """
    mask = mask.resize((image_width, image_height), Image.NEAREST)
    mask = np.array(mask).astype(np.int64)

    mask = mask - 1  
    mask = torch.from_numpy(mask)
    return mask


def prepare_data(batch_size: int, image_width: int, image_height: int):
    transform = transforms.Compose([
        transforms.Resize((image_width, image_height)),
        transforms.ToTensor()
    ])

    dataset = datasets.OxfordIIITPet(
        root="data",
        download=True,
        target_types="segmentation",
        transform=transform,
        target_transform=lambda mask: mask_to_long(mask, image_width, image_height)
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_unet(dataloader, epochs, model_weights, lr, is_plus, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA доступна: {"Да" if torch.cuda.is_available() else "Нет"}")
    print(f"Текущее устройство: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}")
    
    if is_plus:
        model = UNetPlus(in_channels=3, out_channels=3, init_features=settings.INIT_FEATURES)
    else:
        model = UNet(in_channels=3, out_channels=3, init_features=settings.INIT_FEATURES)
    if model_weights:
        model.load_state_dict(model_weights)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, masks in tqdm(dataloader, desc=f"Эпоха {epoch+1}/{epochs}"):
            images = images.to(device)
            masks = masks.to(device)  

            optimizer.zero_grad()
            outputs = model(images) 
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Средняя потеря: {running_loss/len(dataloader):.4f}")

    return model


def visualize_results(model, dataloader, num_samples: int, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_imgs, all_masks, all_preds = [], [], []
    
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            preds_class = torch.argmax(preds, dim=1)
            
            all_imgs.append(imgs.cpu())
            all_masks.append(masks.cpu())
            all_preds.append(preds_class.cpu())
            
            total_collected = sum(img.size(0) for img in all_imgs)
            if total_collected >= num_samples:
                break
    
    imgs = torch.cat(all_imgs)[:num_samples]
    masks = torch.cat(all_masks)[:num_samples]
    preds_class = torch.cat(all_preds)[:num_samples]

    plt.figure(figsize=(15, 3 * num_samples))
    for i in range(num_samples):
        plt.subplot(num_samples, 3, i * 3 + 1)
        plt.imshow(imgs[i].permute(1, 2, 0))
        plt.title("Оригинал")
        plt.axis("off")

        plt.subplot(num_samples, 3, i * 3 + 2)
        plt.imshow(masks[i], cmap="gray")
        plt.title("Истинная маска")
        plt.axis("off")

        plt.subplot(num_samples, 3, i * 3 + 3)
        plt.imshow(preds_class[i], cmap="gray")
        plt.title("Предсказание")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(settings.PLOT_CHECK_SAVE_PATH, dpi=150, bbox_inches='tight')
    print(f"График сохранён в {settings.PLOT_CHECK_SAVE_PATH}")
