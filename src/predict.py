import glob
from src.train_model import fetch_env_dict
import torch
import albumentations
import pandas as pd
import numpy as np
import tqdm
from PIL import Image
import joblib
import torch.nn as nn
from torch.nn import functional as F
from data.dataset import TestBengaliClassificationDataset
from models.model import ResNet18

env_dict = fetch_env_dict()


def model_predict(model):
    c_pred, p_pred, s_pred, l_pred, n_pred, m_pred, f_pred = [], [], [], [], [], [], []
    img_ids_list = []

    df = pd.read_csv("/Users/Banner/Downloads/test.csv")

    dataset = TestBengaliClassificationDataset(
        image_paths, targets, resize, augmentations
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=env_dict["TEST_BATCH_SIZE"],
        shuffle=False,
        num_workers=4,
    )

    for bi, d in enumerate(data_loader):
        image = d["image"]
        img_id = d["img_id"]
        image = image.to(env_dict["DEVICE"], type=torch.float)
        categories, pattern, sleeve, length, neckline, material, fit = model(image)

        for ii, imid in enumerate(img_id):
            c_pred.append(categories[ii].cpu().detach().numpy())
            p_pred.append(pattern[ii].cpu().detach().numpy())
            s_pred.append(sleeve[ii].cpu().detach().numpy())
            l_pred.append(length[ii].cpu().detach().numpy())
            n_pred.append(neckline[ii].cpu().detach().numpy())
            m_pred.append(material[ii].cpu().detach().numpy())
            f_pred.append(fit[ii].cpu().detach().numpy())
            img_ids_list.append(imid)

    return c_pred, p_pred, s_pred, l_pred, n_pred, m_pred, f_pred, img_ids_list


def main():
    model = ResNet18(pretrained=False)
    final_c_pred = []
    final_p_pred = []
    final_s_pred = []
    final_l_pred = []
    final_n_pred = []
    final_m_pred = []
    final_f_pred = []
    final_img_ids = []

    for i in range(5):
        model.load_state_dict(torch.load(f"../models/resnet18_fold{i}.bin"))
        model.to(fetch_env_dict["DEVICE"])
        model.eval()
        (
            c_pred,
            p_pred,
            s_pred,
            l_pred,
            n_pred,
            m_pred,
            f_pred,
            img_ids_list,
        ) = model_predict()
        final_c_pred.append(c_pred)
        final_p_pred.append(p_pred)
        final_s_pred.append(s_pred)
        final_l_pred.append(l_pred)
        final_n_pred.append(n_pred)
        final_m_pred.append(m_pred)
        final_f_pred.append(f_pred)
        if i == 0:
            final_img_ids.extend(img_ids_list)
        final_c = np.argmax(np.mean(np.array(final_c_pred), axis=0), axis=1)
        final_p = np.argmax(np.mean(np.array(final_p_pred), axis=0), axis=1)
        final_s = np.argmax(np.mean(np.array(final_s_pred), axis=0), axis=1)
        final_l = np.argmax(np.mean(np.array(final_l_pred), axis=0), axis=1)
        final_n = np.argmax(np.mean(np.array(final_n_pred), axis=0), axis=1)
        final_m = np.argmax(np.mean(np.array(final_m_pred), axis=0), axis=1)
        final_f = np.argmax(np.mean(np.array(final_f_pred), axis=0), axis=1)

    predictions = []
    for ii, imid in enumerate(final_img_ids):
        predictions.append((f"{imid}_category", final_c))
        predictions.append((f"{imid}_pattern", final_p))
        predictions.append((f"{imid}_sleeve", final_s))
        predictions.append((f"{imid}_length", final_l))
        predictions.append((f"{imid}_neckline", final_n))
        predictions.append((f"{imid}_material", final_m))
        predictions.append((f"{imid}_fit", final_f))

    submission = pd.DataFrame(predictions, columns=["row_id", "target"])
    submission.to_csv("submission", index=False)
