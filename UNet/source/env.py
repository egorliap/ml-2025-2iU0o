from dataclasses import dataclass

@dataclass
class Settings:
    INIT_FEATURES=64
    MODEL_SAVE_PATH="./model/unet_model"
    PLOT_CHECK_SAVE_PATH="results/plot_train_result.png"
    DEFAULT_IMAGE_WIDTH=256
    DEFAULT_IMAGE_HEIGHT=256

settings = Settings()