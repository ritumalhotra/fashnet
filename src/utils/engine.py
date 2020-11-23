import tqdm
import torch

import torch.nn as nn


def loss_fn(outputs, targets):
    return nn.CrossEntropyLoss(outputs, targets)


def train(dataset, data_loader, env_dict, model, optimizer):
    model.train()
    for bi, d in tqdm(
        enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)
    ):
        image = d["image"]
        categories = d["categories"]
        pattern = d["pattern"]
        sleeve = d["sleeve"]
        length = d["length"]
        neckline = d["neckline"]
        material = d["material"]
        fit = d["fit"]

        image = image.to(env_dict["DEVICE"], dtype=torch.float)
        categories = categories.to(env_dict["DEVICE"], dtype=torch.float)
        pattern = pattern.to(env_dict["DEVICE"], dtype=torch.float)
        sleeve = sleeve.to(env_dict["DEVICE"], dtype=torch.float)
        length = length.to(env_dict["DEVICE"], dtype=torch.float)
        neckline = neckline.to(env_dict["DEVICE"], dtype=torch.float)
        material = material.to(env_dict["DEVICE"], dtype=torch.float)
        fit = fit.to(env_dict["DEVICE"], dtype=torch.float)

        optimizer.zero_grad()

        outputs = model(image)
        targets = (categories, pattern, sleeve, length, neckline, material, fit)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(dataset, data_loader, env_dict, model):
    model.eval()

    final_loss = 0
    counter = 0

    for bi, d in tqdm(
        enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)
    ):
        counter += 1
        image = d["image"]
        categories = d["categories"]
        pattern = d["pattern"]
        sleeve = d["sleeve"]
        length = d["length"]
        neckline = d["neckline"]
        material = d["material"]
        fit = d["fit"]

        image = image.to(env_dict["DEVICE"], dtype=torch.float)
        categories = categories.to(env_dict["DEVICE"], dtype=torch.float)
        pattern = pattern.to(env_dict["DEVICE"], dtype=torch.float)
        sleeve = sleeve.to(env_dict["DEVICE"], dtype=torch.float)
        length = length.to(env_dict["DEVICE"], dtype=torch.float)
        neckline = neckline.to(env_dict["DEVICE"], dtype=torch.float)
        material = material.to(env_dict["DEVICE"], dtype=torch.float)
        fit = fit.to(env_dict["DEVICE"], dtype=torch.float)

        outputs = model(image)
        targets = (categories, pattern, sleeve, length, neckline, material, fit)
        loss = loss_fn(outputs, targets)
        final_loss += loss

    return final_loss / counter
