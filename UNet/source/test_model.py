import os
import glob
import argparse
from dataclasses import dataclass

import torch
from torchinfo import summary
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from model import UNet 
from plus_model import UNetPlus
from env import settings


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    # объект: черный, фон: серый, граница: белый
    out[mask == 0] = [0, 0, 0]
    out[mask == 1] = [128, 128, 128]
    out[mask == 2] = [255, 255, 255]
    return out


def load_model(model_path, is_plus, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_plus:
        model = UNetPlus(in_channels=3, out_channels=3, init_features=settings.INIT_FEATURES).to(device)
    else:
        model = UNet(in_channels=3, out_channels=3, init_features=settings.INIT_FEATURES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("Модель успешно загружена")
    return model, device

def crop_image(img: Image.Image, target_ratio) -> Image.Image:
    width, height = img.size
    current_ratio = width / height

    if current_ratio > target_ratio:
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        right = left + new_width
        top, bottom = 0, height
    else:
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        bottom = top + new_height
        left, right = 0, width

    return img.crop((left, top, right, bottom))

def preprocess_image(image_path, model_width: int, model_height: int):
    img = Image.open(image_path).convert("RGB")
    img = crop_image(img, settings.DEFAULT_IMAGE_WIDTH / settings.DEFAULT_IMAGE_HEIGHT)
    img_resized = img.resize((model_width, model_height), Image.Resampling.LANCZOS)
    transform = transforms.ToTensor()
    img_tensor = transform(img_resized).unsqueeze(0) 
    return img_tensor, img

def predict_mask(model, image_tensor, device):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor) 
        preds_class = torch.argmax(output, dim=1).cpu().squeeze(0)  
        return preds_class.numpy()

def visualize_result(original_image: Image.Image, mask: np.ndarray, plot_save_path):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Исходное изображение")
    plt.imshow(original_image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Предсказанная маска")
    mask_colored = colorize_mask(mask)
    plt.imshow(mask_colored)
    plt.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_save_path), exist_ok=True)
    plt.savefig(plot_save_path)
    print(f"График сохранён в {plot_save_path}")

def get_result_path(image_path, results_dir="./results"):
    base_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_name)[0]
    return os.path.join(results_dir, f"plot_{name_without_ext}.png")

@dataclass
class CLA:
    model_postfix: str

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_postfix", type=str, default="", help=f"Program will search for model named {settings.MODEL_SAVE_PATH}<model_prefix>.pth")
    cla: CLA = parser.parse_args()

    model, device = load_model(f"{settings.MODEL_SAVE_PATH}_{cla.model_postfix}.pth", True)
    summary(model)

    test_images = glob.glob("./data/custom_images/*")

    for img_path in test_images:
        img_tensor, orig_img = preprocess_image(img_path, settings.DEFAULT_IMAGE_WIDTH, settings.DEFAULT_IMAGE_HEIGHT)
        mask = predict_mask(model, img_tensor, device)
        visualize_result(orig_img, mask, get_result_path(img_path))

if __name__ == "__main__":
    main()
