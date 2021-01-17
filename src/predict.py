import glob
import os
import ast
import torch
import albumentations
import pandas as pd
import numpy as np
import tqdm
from PIL import Image
import joblib
import torch.nn as nn
from torch.nn import functional as F
import albumentations as A
from data.dataset import TestClassificationDataset
from models.model import ResNet18


def fetch_env_dict():
    env_dict = {}
    env_dict["DEVICE"] = os.environ.get("DEVICE")
    env_dict["IMG_HEIGHT"] = int(os.environ.get("IMG_HEIGHT"))
    env_dict["IMG_WIDTH"] = int(os.environ.get("IMG_WIDTH"))
    env_dict["TEST_BATCH_SIZE"] = int(os.environ.get("TEST_BATCH_SIZE"))
    env_dict["MODEL_MEAN"] = ast.literal_eval(os.environ.get("MODEL_MEAN"))
    env_dict["MODEL_STD"] = ast.literal_eval(os.environ.get("MODEL_STD"))

    return env_dict


def model_predict(model, env_dict):
    c_pred, p_pred, s_pred, l_pred, n_pred, m_pred, f_pred = [], [], [], [], [], [], []
    img_ids_list = []

    # TODO(Sayar) Handle path better
    df = pd.read_csv("/Users/Banner/Downloads/test.csv")

    aug = A.Compose(
        [
            A.Normalize(
                env_dict["MODEL_MEAN"],
                env_dict["MODEL_STD"],
                max_pixel_value=255.0,
                always_apply=True,
            )
        ]
    )

    dataset = TestClassificationDataset(
        image_paths=df["img_path"].tolist(),
        resize=(env_dict["IMG_HEIGHT"], env_dict["IMG_WIDTH"]),
        augmentations=aug,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=env_dict["TEST_BATCH_SIZE"],
        shuffle=False,
        num_workers=4,
    )

    for bi, d in enumerate(data_loader):
        image = d["image"]
        img_id = d["image_id"]
        image = image.to(env_dict["DEVICE"], dtype=torch.float)
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
    env_dict = fetch_env_dict()
    final_c_pred = []
    final_p_pred = []
    final_s_pred = []
    final_l_pred = []
    final_n_pred = []
    final_m_pred = []
    final_f_pred = []
    final_img_ids = []

    for i in range(5):
        # TODO(Sayar): Handle CUDA vs CPU flag
        model.load_state_dict(
            torch.load(
                f"../models/resnet18_fold{i}.bin",
                map_location=torch.device(env_dict["DEVICE"]),
            )
        )
        model.to(env_dict["DEVICE"])
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
        ) = model_predict(model, env_dict)
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
        predictions.append((f"{imid}_fit", final_c[ii]))
        predictions.append((f"{imid}_pattern", final_p[ii]))
        predictions.append((f"{imid}_sleeve", final_s[ii]))
        predictions.append((f"{imid}_length", final_l[ii]))
        predictions.append((f"{imid}_neckline", final_n[ii]))
        predictions.append((f"{imid}_material", final_m[ii]))
        predictions.append((f"{imid}_category", final_f[ii]))

    submission = pd.DataFrame(predictions, columns=["row_id", "target"])
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
