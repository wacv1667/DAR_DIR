import gc
import torch
from tqdm import tqdm
import torch.nn.functional as F
from bayesian_torch.models.dnn_to_bnn import get_kl_loss


def train_loop(model, train_loader,
               criterion, optimizer,
               accelerator, scheduler=None,
               verbose=False, 
               bbb = False):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
                 
    for i, x in (enumerate(train_loader)):
        with accelerator.accumulate(model):
            x = {k:v.to(accelerator.device) for (k,v) in x.items()}
            targets = x["label"]
            outputs = model(x)
            loss = criterion(outputs, targets)
            if bbb: 
                kl = get_kl_loss(model)
                loss = loss + kl / targets.size(0) # divide KL by the batch_size 
            accelerator.backward(loss)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
            
            # Compute accuracy and accumulate loss per batch
            total_loss += loss.item() 
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == torch.argmax(targets, dim=1)).sum().item() 

    # Compute epoch accuracy and loss
    accuracy = correct_predictions / total_predictions
    epoch_loss = total_loss / (i+1)
    gc.collect(), torch.cuda.empty_cache()
    if verbose:
        print(f"Train Accuracy: {accuracy:.4f}")
        print(f"Train Loss: {epoch_loss:.4f}") 
    return model, optimizer, scheduler, accuracy, epoch_loss
