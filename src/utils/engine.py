
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

        # TODO(Sayar) Add target value mappings
        image = image.to(env_dict["DEVICE"], dtype=torch.float)
        optimizer.zero_grad()

        outputs = model(image)
        targets = ()
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimizer.step()


def evaluate(data_loader, model, device):
    model.eval()

    final_targets = []
    final_outputs = []

    with torch.no_grad():

        for data in data_loader:
            inputs = data["image"]
            targets = data["targets"]
            inputs = inputs.to(device, dtype=torch.float)
            targets = inputs.to(device, dtype=torch.float)

            output = model(input)

            targets = targets.detach().cpu().numpy().tolist()
            output = output.detach().cpu().numpy().tolist()

            final_targets.extend(targets)
            final_outputs.extend(output)

    return final_outputs, final_targets