import ast
import os
import torch.nn as nn

def fetch_env_vars():
    env_dict = {}
    env_dict["DEVICE"] = os.environ.get("DEVICE")
    env_dict["TRAINING_FOLDS_CSV"] = os.environ.get("TRAINING_FOLDS_CSV")
    env_dict["IMG_HEIGHT"] = int(os.environ.get("IMG_HEIGHT"))
    env_dict["IMG_WIDTH"]  = int(os.environ.get("IMG_WIDTH"))
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