from __future__ import absolute_import, division, print_function
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
import os
from fSRN import fSRN
from options import save_options
from utility_functions import make_coord_grid, create_folder

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

# Saves the model in compressed form (lossless)
def save_model(model,opt):
    folder = create_folder(save_folder, opt["save_name"])
    path_to_save = os.path.join(save_folder, folder)
    
    torch.save({'state_dict': model.state_dict()}, 
        os.path.join(path_to_save, "model.ckpt.tar"),
        pickle_protocol=4
    )
    save_options(opt, path_to_save)

# Loads a model
def load_model(opt, device):
    path_to_load = os.path.join(save_folder, opt["save_name"])
    model = create_model(opt)

    ckpt = torch.load(os.path.join(path_to_load, 'model.ckpt.tar'), 
        map_location = device)
    
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(opt['device'])
    return model

def create_model(opt):
    if(opt['model'] == 'fSRN'):
        return fSRN(opt)
    else:
        print(f"Model {opt['model']} is not implemented.")
        quit()

# Samples a (implicit) network on a grid and returns a 
# volume of the result, querying at max max_points per
# forward pass to reduce memory cost.
def sample_grid(model, grid, max_points = 100000):
    coord_grid = make_coord_grid(grid, 
        model.opt['device'], flatten=False,
        align_corners=model.opt['align_corners'])
    coord_grid_shape = list(coord_grid.shape)
    coord_grid = coord_grid.view(-1, coord_grid.shape[-1])
    coord_grid = coord_grid.requires_grad_(True)
    
    vals = forward_maxpoints(model, coord_grid,
                max_points = max_points)
    coord_grid_shape[-1] = model.opt['n_dims']
    vals = vals.reshape(coord_grid_shape)
    return vals

def sample_grid_potentials(model, grid, max_points = 100000):
    coord_grid = make_coord_grid(grid, 
        model.opt['device'], flatten=False,
        align_corners=model.opt['align_corners'])
    
    final_sp_output_shape = list(coord_grid.shape)
    final_vp_output_shape = list(coord_grid.shape)
    final_sp_output_shape[-1] = 1
    final_vp_output_shape[-1] = 3
    
    coord_grid = coord_grid.view(-1, coord_grid.shape[-1])
    coord_grid = coord_grid.requires_grad_(True)
    
    sp_output_shape = list(coord_grid.shape)
    vp_output_shape = list(coord_grid.shape)
    sp_output_shape[-1] = 1
    vp_output_shape[-1] = 3
    sp_output = torch.empty(sp_output_shape, 
        dtype=torch.float32, device=model.opt['device'])
    vp_output = torch.empty(vp_output_shape, 
        dtype=torch.float32, device=model.opt['device'])

    for start in range(0, coord_grid.shape[0], max_points):
        #print("%i:%i" % (start, min(start+max_points, coords.shape[0])))
        sp, vp = model.forward_potentials(
            coord_grid[start:min(start+max_points, coord_grid.shape[0])]
            )
        sp = sp.detach()
        vp = vp.detach()
        
        sp_output[start:min(start+max_points, coord_grid.shape[0])] = \
            sp.unsqueeze(1)
        vp_output[start:min(start+max_points, coord_grid.shape[0])] = \
            vp
        model.zero_grad()

    sp_output = sp_output.reshape(final_sp_output_shape)
    vp_output = vp_output.reshape(final_vp_output_shape)
    return sp_output, vp_output

def sample_grid_vectorfields(model, grid, max_points = 100000):
    coord_grid = make_coord_grid(grid, 
        model.opt['device'], flatten=False,
        align_corners=model.opt['align_corners'])
    final_shape = list(coord_grid.shape)
    final_shape[-1] = 3
    coord_grid = coord_grid.view(-1, coord_grid.shape[-1])
    coord_grid = coord_grid.requires_grad_(True)
    
    rotationfree_output_shape = list(coord_grid.shape)
    divergencefree_output_shape = list(coord_grid.shape)
    rotationfree_output_shape[-1] = 3
    divergencefree_output_shape[-1] = 3
    rotationfree_output = torch.empty(rotationfree_output_shape, 
        dtype=torch.float32, device=model.opt['device'])
    divergencefree_output = torch.empty(divergencefree_output_shape, 
        dtype=torch.float32, device=model.opt['device'])
    
    for start in range(0, coord_grid.shape[0], max_points):
        #print("%i:%i" % (start, min(start+max_points, coords.shape[0])))
        rotationfree, divergencefree = model.forward_vectorfields(
            coord_grid[start:min(start+max_points, coord_grid.shape[0])]
            )
        rotationfree = rotationfree.detach()
        divergencefree = divergencefree.detach()
        rotationfree_output[start:min(start+max_points, coord_grid.shape[0])] = \
            rotationfree
        divergencefree_output[start:min(start+max_points, coord_grid.shape[0])] = \
            divergencefree
        model.zero_grad()

    rotationfree_output = rotationfree_output.reshape(final_shape)
    divergencefree_output = divergencefree_output.reshape(final_shape)
    return rotationfree_output, divergencefree_output

# Forward for the model with built in
# filtering to only process max_points at a time 
# maximum
def forward_maxpoints(model, coords, max_points=100000):
  
    output_shape = list(coords.shape)
    output_shape[-1] = model.opt['n_dims']
    output = torch.empty(output_shape, 
        dtype=torch.float32, device=model.opt['device'])
    #print(output.shape)
    for start in range(0, coords.shape[0], max_points):
        #print("%i:%i" % (start, min(start+max_points, coords.shape[0])))
        output[start:min(start+max_points, coords.shape[0])] = \
            model(coords[start:min(start+max_points, coords.shape[0])]).detach()
        model.zero_grad()
    return output

class LReLULayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False):
        super().__init__()
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, 
            bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(
                -torch.nn.init.calculate_gain("leaky_relu", 0.2),
                torch.nn.init.calculate_gain("leaky_relu", 0.2)
            )

    def forward(self, input):
        return F.leaky_relu(self.linear(input), 0.2)
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.linear(input)
        return F.leaky_relu(intermediate, 0.2), intermediate

