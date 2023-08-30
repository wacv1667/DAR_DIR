import os
from glob import glob
import pandas as pd


def make_results_folder(base_path:str="results"):
    try:
        os.mkdir(base_path)
    except FileExistsError:
        pass
    exp_num = len(glob(f"{base_path}/*")) + 1 
    folder_path = f"{base_path}/{exp_num}"
    os.mkdir(folder_path)
    return folder_path

def store_training_results(folder, epoch, train_loss, 
                  train_acc, val_loss, val_acc):
    try:
        df = pd.read_csv(f"{folder}/results.csv")
    except Exception as e:
        df = pd.DataFrame(columns=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    df.loc[len(df)] = [epoch, train_loss, train_acc, val_loss, val_acc]
    df.to_csv(f"{folder}/results.csv", index=False) 

def freeze_params(model):
    # Freeze all parameters of the model
    for param in model.parameters():
        param.requires_grad = False

    # Print to verify that all parameters are frozen
    for name, param in model.named_parameters():
        if "attention" in name:
            param.requires_grad = True
        if "classifier" in name:
            param.requires_grad =True

def load_two_stream_model(train_ds):
    model = TwoStreamAttentionFusion(train_ds)
    freeze_params(model.inside_vmae)
    freeze_params(model.outside_vmae)
#     for name, param in model.named_parameters():
#         print(f'{name}: requires_grad={param.requires_grad}')
    return model
