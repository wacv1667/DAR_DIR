import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from math import ceil
from accelerate import Accelerator

import os
from glob import glob

from .training import train_loop, eval_loop
from .tools import make_results_folder, store_results


def run_training(
        batch_size,
        train_loader,
        val_loader,
        model,
        desired_bs=8,
        num_epochs=200,
        lr=5e-5,
        wd=0.05,
        warmup_steps=100,
        criterion=torch.nn.CrossEntropyLoss(label_smoothing=0.0),
        early_stopping=25,
        verbose:bool=False,
        bbb=False,):
    
    # create resuls folder
    folder_path = make_results_folder()
    
    # setup torch params
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    optimizer = AdamW(params=model.parameters(), lr=lr, weight_decay=wd)
    gas = ceil(desired_bs//batch_size)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=(len(train_loader) * num_epochs // gas),
    )
    
    accelerator = Accelerator(gradient_accumulation_steps=gas)
    device = accelerator.device
    model = model.to(device)
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, lr_scheduler)
        
    for epoch in range(1,num_epochs+1):
        model, optimizer, lr_scheduler, train_acc, train_loss  = train_loop(model, train_loader, 
                                                                            criterion, optimizer, 
                                                                            accelerator, scheduler=lr_scheduler, 
                                                                            verbose=verbose, bbb=bbb)     
        model = model.to(device)
        val_result_dict =  eval_loop(model, val_loader, criterion, verbose=verbose)        
        store_results(folder_path, epoch, train_loss, train_acc, val_result_dict["loss"], val_result_dict["accuracy"])

        if (epoch == 1) or (val_result_dict["loss"] < best_loss):
            best_loss =  val_result_dict["loss"] 
            print(f"New best val loss: Epoch #{str(epoch)}: {round(best_loss, 4)}")
            early_stopping_count = 0
            model_path =  f"{folder_path}/best_model.pth"
            torch.save(model, model_path)
        else:
            early_stopping_count += 1
            if early_stopping_count == early_stopping:
                break
