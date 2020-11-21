import ast
import os

from numpy.core.fromnumeric import resize
from src.data.dataset import ClassificationDataLoader, ClassificationDataset
from src.models.model_dispatcher import MODEL_DISPATCHER
import torch
import torch.nn as nn
import tqdm
import albumentations as A


def fetch_env_dict():
    env_dict = {}
    env_dict["DEVICE"] = os.environ.get("DEVICE")
    env_dict["TRAINING_FOLDS_CSV"] = os.environ.get("TRAINING_FOLDS_CSV")
    env_dict["IMG_HEIGHT"] = int(os.environ.get("IMG_HEIGHT"))
    env_dict["IMG_WIDTH"] = int(os.environ.get("IMG_WIDTH"))
    env_dict["EPOCHS"] = int(os.environ.get("EPOCHS"))

    env_dict["TRAIN_BATCH_SIZE"] = int(os.environ.get("TRAIN_BATCH_SIZE"))
    env_dict["TEST_BATCH_SIZE"] = int(os.environ.get("TEST_BATCH_SIZE"))

    env_dict["MODEL_MEAN"] = ast.literal_eval(os.environ.get("MODEL_MEAN"))
    env_dict["MODEL_STD"] = ast.literal_eval(os.environ.get("MODEL_STD"))

    env_dict["TRAINING_FOLDS"] = ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
    env_dict["VALIDATION_FOLDS"] = ast.literal_eval(os.environ.get("VALIDATION_FOLDS"))

    env_dict["BASE_MODEL"] = os.environ.get("BASE_MODEL")

    return env_dict


def loss_fn(outputs, targets):
    return nn.CrossEntropyLoss(outputs, targets)


def train(dataset, data_loader, env_dict, model, optimizer):
    model.train()
    for bi, d in tqdm(
        enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)
    ):
        image = d["image"]

        # TODO(Sayar) Add target value mappings
        image = image.to(env_dict["DEVICE"], dtype=torch.float)
        optimizer.zero_grad()

        outputs = model(image)
        targets = ()
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()


def main():
    env_dict = fetch_env_dict()
    model = MODEL_DISPATCHER[env_dict["BASE_MODEL"]](pretrained=True)
    model.to(env_dict["DEVICE"])

    aug = A.Compose(
        [
            A.Resize(200, 300),
            A.CenterCrop(100, 100),
            A.RandomCrop(80, 80),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=(-90, 90)),
            A.VerticalFlip(p=0.5),
            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = ClassificationDataset(
        image_paths=image_paths,
        targets=targets,
        resize=targets,
        augmentations=aug,
    )

    #TODO(Sayar): Add parameters for dataloader
    train_data_loader = ClassificationDataLoader(
        train_dataset, image_paths=None, targets=None, resize=None, augmentations=None
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.4, verbose=True
    )
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    for epoch in range(env_dict["EPOCHS"]):
        train(train_dataset, train_data_loader, env_dict, model, optimizer)
        # TODO(Sayar): Add evaluation dataset, dataloader
        val_score = evaluate(valid_dataset, valid_data_loader, model)
        scheduler.step(val_score)
        torch.save(
            model.state_dict(),
            f"{env_dict['BASE_MODEL']}_fold{env_dict['VALIDATION_FOLDS'][0]}.bin",
        )

if __name__ == "__main__":
    main()
