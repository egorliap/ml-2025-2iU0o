import argparse
from dataclasses import dataclass
import torch

from train_utils import prepare_data, train_unet, visualize_results
from env import settings

@dataclass
class CLA:
    batch: int
    epochs: int
    model_postfix: str
    model_save_postfix: str
    lr: float

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=str, required=False, help="Batch Size", default=8)
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_postfix", type=str, default="", help=f"Program will search for model named {settings.MODEL_SAVE_PATH}<model_prefix>.pth")
    parser.add_argument("--model_save_postfix", type=str, default="", help=f"Program will save model as {settings.MODEL_SAVE_PATH}<model_save_postfix>.pth")
    cla: CLA = parser.parse_args()

    print(cla)

    model_path: str = f"{settings.MODEL_SAVE_PATH}_{cla.model_postfix}.pth"
    model_weights = None
    
    try:
        model_weights = torch.load(model_path, weights_only=True)
        print(f"Файл {model_path} найден, дообучение модели")
    except FileNotFoundError:
        print(f"Файл {model_path} не найден, обучение будет с нуля")

    print("Подготовка данных...")
    dataloader = prepare_data(batch_size=cla.batch, image_width=settings.DEFAULT_IMAGE_WIDTH, image_height=settings.DEFAULT_IMAGE_HEIGHT)

    print("Обучение модели...")
    model = train_unet(dataloader, epochs=cla.epochs, model_weights=model_weights, lr=cla.lr, is_plus=True)

    model_save_path: str = f"{settings.MODEL_SAVE_PATH}_{cla.model_save_postfix}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Модель сохранена в {model_save_path}")

    print("Визуализация результатов...")
    visualize_results(model, dataloader, num_samples=10)

if __name__ == "__main__":
    main()