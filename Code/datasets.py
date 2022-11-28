import os
from utility_functions import make_coord_grid, nc_to_tensor, curl
import torch
import torch.nn.functional as F

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        
        self.opt = opt
        self.min_ = None
        self.max_ = None
        self.mean_ = None
        self.full_coord_grid = None
        folder_to_load = os.path.join(data_folder, self.opt['data'])

        # Load data
        print(f"Initializing dataset - reading {folder_to_load}")
        d = nc_to_tensor(folder_to_load).to(opt['data_device'])

        # Compute vorticity field if requested
        if(opt['vorticity']):
            print(f"Using vorticity field")
            d = curl(d)
            #tensor_to_cdf(d, "vorticity.nc")

        # normalize by dividing my max length
        # d /= (d.norm(dim=1).max() + 1e-8)
        self.data = d
        print("Data size: " + str(self.data.shape))
            
        # Generate the coordinate grid [n, num_dims]
        # which are coordinates in the domain [-1, 1]
        # for each dimension which match the dimensions
        # of the vector field's size
        self.index_grid = make_coord_grid(
            self.data.shape[2:], 
            self.opt['data_device'],
            flatten=True,
            align_corners=self.opt['align_corners'])
        
        # print some statistics
        print("Min/mean/max: %0.04f, %0.04f, %0.04f" % \
            (self.min(), self.mean(), self.max()))
        print("Min/mean/max mag: %0.04f, %0.04f, %0.04f" % \
            (self.data.norm(dim=1).min(), 
            self.data.norm(dim=1).mean(), 
            self.data.norm(dim=1).max()))          

    # min, max, and mean functions which cache
    # the result for improved speed if requested
    # multiple times
    def min(self):
        if self.min_ is not None:
            return self.min_
        else:
            self.min_ = self.data.min()
            return self.min_
    def mean(self):
        if self.mean_ is not None:
            return self.mean_
        else:
            self.mean_ = self.data.mean()
            return self.mean_
    def max(self):
        if self.max_ is not None:
            return self.max_
        else:
            self.max_ = self.data.max()
            return self.max_

    # takes the middle x-axis slice of the data
    def get_2D_slice(self):
        if(len(self.data.shape) == 4):
            return self.data[0].clone()
        else:
            return self.data[0,:,:,:,int(self.data.shape[4]/2)].clone()

    # total number of points in the data
    def total_points(self):
        t = 1
        for i in range(2, len(self.data.shape)):
            t *= self.data.shape[i]
        return t

    # Creates the coordinate grid for the shape of
    # the data.
    def get_full_coord_grid(self):
        if self.full_coord_grid is None:
            self.full_coord_grid = make_coord_grid(self.data.shape[2:], 
                    self.opt['data_device'], flatten=True, 
                    align_corners=self.opt['align_corners'])
        return self.full_coord_grid
        
    # samples n_points points at random from
    # the vector field, along with the positions
    # returns x, y where 
    # x is a [n, n_dims] list of coordinates
    # y is a [n, n_dims] list of vector values at the corresponding locations
    def get_random_points(self, n_points):        
        possible_spots = self.index_grid
        
        # If interpolation between points is enabled
        if(self.opt['interpolate']):
            x = torch.rand([1, 1, 1, n_points, self.opt['n_dims']], 
                device=self.opt['data_device']) * 2 - 1
            # 'mode' is bilinear regardless of actual dimensionality
            y = F.grid_sample(self.data,
                x, mode='bilinear', 
                align_corners=self.opt['align_corners'])
        else:
            if(n_points >= possible_spots.shape[0]):
                x = possible_spots.clone().unsqueeze_(0)
            else:
                samples = torch.randperm(possible_spots.shape[0], 
                    dtype=torch.long, device=self.opt['data_device'])[:n_points]
                # Change above to not use CPU when not on MPS
                # Verify that the bottom two lines do the same thing
                x = torch.index_select(possible_spots, 0, samples).clone().unsqueeze_(0)
                #x = possible_spots[samples].clone().unsqueeze_(0)
            for _ in range(len(self.data.shape[2:])-1):
                x = x.unsqueeze(-2)
            
            y = F.grid_sample(self.data, 
                x, mode='nearest', 
                align_corners=self.opt['align_corners'])
        
        y = y.squeeze()
        if(len(y.shape) == 1):
            y = y.unsqueeze(0)    
        
        y = y.permute(1,0)              
        x = x.squeeze()
        
        return x, y

def generate_synthetic_field1():
    # Uses the magnitude of the velocity field from the tornado dataset as a scalar
    # potential, and the vector field from the ABC dataset as a vector potential
    # in a HHD. Adds the gradient/curl fields together to generate the
    # final vector field.

    tornado = nc_to_tensor(os.path.join(data_folder, "tornado.nc"))
    abc = nc_to_tensor(os.path.join(data_folder, "ABC.nc"))

    tornado_norm = torch.norm(tornado, dim=1, keepdim=True)

    from utility_functions import spatial_gradient, tensor_to_cdf
    rotation_free_u = spatial_gradient(tornado_norm, 0, 0)
    rotation_free_v = spatial_gradient(tornado_norm, 0, 1)
    rotation_free_w = spatial_gradient(tornado_norm, 0, 2)

    rotation_free = torch.cat([rotation_free_u, rotation_free_v, rotation_free_w], dim=1)
    divergence_free = curl(abc)

    full_vf = rotation_free + divergence_free

    tensor_to_cdf(full_vf, os.path.join(data_folder, "synthetic_vf.nc"))