import torch
import torch.nn as nn
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm

from model import UNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

'''
Image preprocessing with "albumentations"
    albumentations::Fast image augmentation library and an easy-to-use wrapper around other libraries
    https://github.com/albumentations-team/albumentations

'''

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3 # 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train/"
TRAIN_MASK_DIR = "data/train_masks/"

VAL_IMG_DIR = "data/train/"
VAL_MASK_DIR = "data/train_masks/"

# VAL_IMG_DIR = "data/test/"
# VAL_MASK_DIR = "data/test_masks/"

def train_func(loader, model, optimizer, loss_func, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_func(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    '''
    torch.nn.CrossEntropyLoss()

    torch.nn.BCELoss()
        Creates a criterion that measures the Binary Cross Entropy 
        between the target and the output
    
    torch.nn.BCEWithLogitsLoss()
        Sigmoid + BCELoss

        This loss combines a Sigmoid layer and the BCELoss in one single class. 
        This version is more numerically stable than using a plain Sigmoid 
        followed by a BCELoss as, by combining the operations into one layer, 
        we take advantage of the log-sum-exp trick for numerical stability.
    '''
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # --- Multiple case --- 
    # n = 3
    # model = UNet(in_channels=3, out_channels=N).to(DEVICE)
    # loss_func = nn.CrossEntropyLoss()

    # --- Binary case --- 
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_func = nn.BCEWithLogitsLoss()


    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR,
        VAL_IMG_DIR, VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform, val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("unet_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_func(train_loader, model, optimizer, loss_func, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )

if __name__ == "__main__":
    main()