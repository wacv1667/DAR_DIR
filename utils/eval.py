import gc
import torch
from tqdm import tqdm
import torch.nn.functional as F

def train_eval_loop(model, val_loader, criterion, device="cuda:0", verbose=False):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for i, x in (enumerate(val_loader)):
        x = {k:v.to(device) for (k,v) in x.items()}
        targets = x["label"]
        outputs = model(x)
        val_loss = criterion(outputs, targets)
        
        # Compute accuracy and accumulate loss per batch
        total_loss += val_loss.item() 
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += targets.size(0)
        correct_predictions += (predicted == torch.argmax(targets, dim=1)).sum().item() 

    # Compute epoch accuracy and loss
    accuracy = correct_predictions / total_predictions
    epoch_loss = total_loss / (i+1)
    if verbose:
        print(f"Val Accuracy: {accuracy:.4f}")
        print(f"Val Loss: {epoch_loss:.4f}")  
    return accuracy, epoch_loss
