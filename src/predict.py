import glob
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
from models.model import ResNet34


def model_predict(model):
    c_pred, p_pred, s_pred, l_pred, n_pred, m_pred, f_pred = [], [], [], [], [], [], []
    img_ids_list = []

    df = pd.read_csv("/Users/Banner/Downloads/test.csv")

    dataset = TestBengaliClassificationDataset(image_paths, 
                                               targets, 
                                               resize, 
                                               augmentations)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=4)
    
    for bi, d in enumerate(data_loader):
        image = d["image"]
        img_id = d["img_id"]
        image = image.to(DEVICE, type=torch.float)
        categories, pattern, sleeve, length, neckline, material, fit = model(image)

        for ii, imid in enumerate(img_id):
            c_pred.append(category[ii].cpu().detach().numpy())
            p_pred.append(pattern[ii].cpu().detach().numpy())
            s_pred.append(sleeve[ii].cpu().detach().numpy())
            l_pred.append(length[ii].cpu().detach().numpy())
            n_pred.append(neckline[ii].cpu().detach().numpy())
            m_pred.append(material[ii].cpu().detach().numpy())
            f_pred.append(fit[ii].cpu().detach().numpy())
            img_ids_list.append(imid)
    
    return  c_pred, p_pred, s_pred, l_pred, n_pred, m_pred, f_pred, img_ids_list

def main():
    model = ResNet18(pretrained=False)
    final_c_pred = []
    final_p_pred = []
    final_s_pred = []
    final_l_pred = []
    final_n_pred = []
    final_m_pred = []
    final_f_pred = []

    for i in range(5):
        model.load_state_dict(torch.load(f"../models/resnet18_fold{i}.bin"))