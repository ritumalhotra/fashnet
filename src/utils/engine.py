from tqdm import tqdm
import torch

import torch.nn as nn


def loss_fn(outputs, targets):
    o1, o2, o3, o4, o5, o6, o7 = outputs
    t1, t2, t3, t4, t5, t6, t7 = targets
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)
    l4 = nn.CrossEntropyLoss()(o4, t4)
    l5 = nn.CrossEntropyLoss()(o5, t5)
    l6 = nn.CrossEntropyLoss()(o6, t6)
    l7 = nn.CrossEntropyLoss()(o7, t7)

    return (l1 + l2 + l3 + l4 + l5 + l6 + l7) / 7


def train(dataset, data_loader, env_dict, model, optimizer):
    model.train()
    for bi, d in tqdm(
        enumerate(data_loader), total=int(len(dataset) / data_loader.batch_size)
    ):
        # TODO(Sayar): iterate instead of adding each var sequentially
        image = d["image"]
        categories = d["categories"]
        pattern = d["pattern"]
        sleeve = d["sleeve"]
        length = d["length"]
        neckline = d["neckline"]
        material = d["material"]
        fit = d["fit"]

        image = image.to(env_dict["DEVICE"], dtype=torch.float)
        categories = categories.to(env_dict["DEVICE"], dtype=torch.long)
        pattern = pattern.to(env_dict["DEVICE"], dtype=torch.long)
        sleeve = sleeve.to(env_dict["DEVICE"], dtype=torch.long)
        length = length.to(env_dict["DEVICE"], dtype=torch.long)
        neckline = neckline.to(env_dict["DEVICE"], dtype=torch.long)
        material = material.to(env_dict["DEVICE"], dtype=torch.long)
        fit = fit.to(env_dict["DEVICE"], dtype=torch.long)

        optimizer.zero_grad()

        outputs = model(image)
        targets = (pattern, sleeve, length, neckline, material, fit, categories)
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
        categories = categories.to(env_dict["DEVICE"], dtype=torch.long)
        pattern = pattern.to(env_dict["DEVICE"], dtype=torch.long)
        sleeve = sleeve.to(env_dict["DEVICE"], dtype=torch.long)
        length = length.to(env_dict["DEVICE"], dtype=torch.long)
        neckline = neckline.to(env_dict["DEVICE"], dtype=torch.long)
        material = material.to(env_dict["DEVICE"], dtype=torch.long)
        fit = fit.to(env_dict["DEVICE"], dtype=torch.long)

        outputs = model(image)
        targets = (pattern, sleeve, length, neckline, material, fit, categories)
        loss = loss_fn(outputs, targets)
        final_loss += loss

    return final_loss / counter
