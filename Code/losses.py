import torch
import torch.nn.functional as F

'''
For all loss functions, it is assumed x is some reconstructed
data and y is the "ground truth".
'''

def l1(x, y):
    return F.l1_loss(x, y)

def l2(x, y):
    return F.mse_loss(x, y)

def angle_loss(x, y):
    angles = (1 - F.cosine_similarity(x, y))
    mask = (y.norm(dim=1) != 0).type(torch.float32).detach()
    weighted_angles = angles * mask
    return weighted_angles.mean()

def mag_loss(x, y):
    x_norm = torch.norm(x,dim=1)
    y_norm = torch.norm(y,dim=1)
    mags = F.mse_loss(x_norm, y_norm)
    return mags

def magangle_loss(x, y):
    mags = mag_loss(x, y)
    angles = angle_loss(x, y)
    return 0.5*mags + 0.5*angles

def get_loss_func(opt):    
    if(opt['loss'] == "l1"):
        return l1
    elif(opt['loss'] == "l2"):
        return l2
    elif(opt['loss'] == "magangle"):
        return magangle_loss
    else:
        print(f"Missing loss function {opt['training_mode']}. Exiting.")
        quit()