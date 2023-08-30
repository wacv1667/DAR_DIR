from .models import TwoStreamAttentionFusion

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
