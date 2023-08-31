import gc
import torch
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import f1_score
from bayesian_torch.utils.util import predictive_entropy
from torchmetrics.functional import average_precision
from typing import Dict

def eval_loop(model,
              val_loader,
              criterion,
              device="cuda:0",
              verbose=False,
              dataset_name = "brain4cars", 
              mc_samples: int = 1, 
              model_type: str = None) -> Dict:
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    y_true, y_pred, y_pred_softmax = [], [], []
    results_dict = {}
                  
    for i, x in (enumerate(val_loader)):
        x = {k:v.to(device) for (k,v) in x.items()}
        targets = x["label"]

        if mc_samples > 1:
            mc_preds = []
            for _ in (range(n_iters)):
                if model_type == "MC":
                    outputs = model(x)
                elif model_type == "BBB":
                    outputs, _ = model(x) # returns outputs and KL 
                
                mc_preds.append(torch.nn.functional.softmax(outputs.cpu(),dim=1).numpy())
            pe = predictive_entropy(np.array(mc_values).squeeze())
            avg_pred = np.array(mc_values).mean(axis=0) # take the mean of the stochastic predictions
            pred_class = avg_pred.argmax() # take the highest class
            y_pred.extend(pred_class)
            if dataset == "HDD":
                y_pred_softmax.extend(avg_pred.tolist())
        

        else:
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.item()) # change to values if batch_size larger than 1
            if dataset == "HDD":
                y_pred_softmax.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().tolist())
            
        val_loss = criterion(outputs, targets)        
        # Compute accuracy and accumulate loss per batch
        total_loss += val_loss.item() 
        
        total_predictions += targets.size(0)
        correct_predictions += (predicted == torch.argmax(targets, dim=1)).sum().item() 
        y_true.extend(torch.argmax(targets, dim=1).item())
        
    # Compute the relevant metrics per dataset
    results_dict["epoch_loss"] = total_loss / (i+1)

    if dataset == "HDD":
        results_dict["AP"] = average_precision(torch.tensor(y_pred_softmax),
                                               torch.tensor(y_true), 
                                               num_classes=11,
                                               average=None)
    if dataset == "brain4cars":
        results_dict["accuracy"] = correct_predictions / total_predictions
        results_dict["F1"] = f1_score(y_true, y_pred, average="macro")

    return result_dict
