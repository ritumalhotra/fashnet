import ast
import os

import pandas as pd
from numpy.core.fromnumeric import resize
import torch
import torch.nn as nn
import tqdm
import albumentations as A

from utils.engine import train, evaluate
from data.dataset import ClassificationDataLoader, ClassificationDataset
from models.model_dispatcher import MODEL_DISPATCHER


def fetch_env_dict():
    env_dict = {}
    env_dict["DEVICE"] = os.environ.get("DEVICE")
    env_dict["IMG_HEIGHT"] = int(os.environ.get("IMG_HEIGHT"))
    env_dict["IMG_WIDTH"] = int(os.environ.get("IMG_WIDTH"))
    env_dict["EPOCHS"] = int(os.environ.get("EPOCHS"))
    env_dict["IMAGE_PATH"] = os.environ.get("IMAGE_PATH")

    env_dict["TRAIN_BATCH_SIZE"] = int(os.environ.get("TRAIN_BATCH_SIZE"))
    env_dict["VALID_BATCH_SIZE"] = int(os.environ.get("VALID_BATCH_SIZE"))

    env_dict["MODEL_MEAN"] = ast.literal_eval(os.environ.get("MODEL_MEAN"))
    env_dict["MODEL_STD"] = ast.literal_eval(os.environ.get("MODEL_STD"))

    env_dict["TRAINING_FOLDS"] = ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
    env_dict["VALIDATION_FOLDS"] = ast.literal_eval(os.environ.get("VALIDATION_FOLDS"))

    env_dict["BASE_MODEL"] = os.environ.get("BASE_MODEL")

    return env_dict


def main():
    env_dict = fetch_env_dict()
    model = MODEL_DISPATCHER[env_dict["BASE_MODEL"]](pretrained=True)
    model.to(env_dict["DEVICE"])

    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    df = pd.read_csv(os.path.join(parent, "data/train_full.csv"))
    # TODO(Sayar): Remove hacky code here
    train_image_paths = df[df["kfold"].isin(env_dict["TRAINING_FOLDS"])][
        "img_path"
    ].values.tolist()
    val_image_paths = df[df["kfold"].isin(env_dict["VALIDATION_FOLDS"])][
        "img_path"
    ].values.tolist()

    train_image_paths = [
        os.path.join(os.path.join(parent, "data"), img_id)
        for img_id in train_image_paths
    ]
    val_image_paths = [
        os.path.join(os.path.join(parent, "data"), img_id) for img_id in val_image_paths
    ]

    targets = {col: df[col].values for col in df.columns.tolist()[1:-1]}

    aug = A.Compose(
        [
            A.Normalize(
                env_dict["MODEL_MEAN"],
                env_dict["MODEL_STD"],
                max_pixel_value=255.0,
                always_apply=True,
            ),
            A.CenterCrop(100, 100),
            A.RandomCrop(80, 80),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=(-90, 90)),
            A.VerticalFlip(p=0.5),
            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = ClassificationDataset(
        image_paths=train_image_paths,
        targets=targets,
        resize=(env_dict["IMG_HEIGHT"], env_dict["IMG_WIDTH"]),
        augmentations=aug,
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=env_dict["TRAIN_BATCH_SIZE"],
        shuffle=True,
        num_workers=4,
    )

    valid_dataset = ClassificationDataset(
        image_paths=val_image_paths,
        targets=targets,
        resize=(env_dict["IMG_HEIGHT"], env_dict["IMG_WIDTH"]),
        augmentations=aug,
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=env_dict["VALID_BATCH_SIZE"],
        shuffle=False,
        num_workers=4,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.4, verbose=True
    )
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    for epoch in range(env_dict["EPOCHS"]):
        train(train_dataset, train_data_loader, env_dict, model, optimizer)
        val_score = evaluate(valid_dataset, valid_data_loader, env_dict, model)
        scheduler.step(val_score)
        print(f"EPOCH: {epoch}, validation error: {val_score}")
        torch.save(
            model.state_dict(),
            os.path.join(
                parent,
                f"models/{env_dict['BASE_MODEL']}_fold{env_dict['VALIDATION_FOLDS'][0]}.bin",
            ),
        )


if __name__ == "__main__":
    main()
