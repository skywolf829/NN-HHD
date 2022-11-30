from venv import create
import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, 
            bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class ResidualSineLayer(nn.Module):
    def __init__(self, features, bias=True, ave_first=False, ave_second=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0

        self.features = features
        self.linear_1 = nn.Linear(features, features, bias=bias)
        self.linear_2 = nn.Linear(features, features, bias=bias)

        self.weight_1 = .5 if ave_first else 1
        self.weight_2 = .5 if ave_second else 1

        self.init_weights()
  

    def init_weights(self):
        with torch.no_grad():
            self.linear_1.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)
            self.linear_2.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
                                           np.sqrt(6 / self.features) / self.omega_0)

    def forward(self, input):
        sine_1 = torch.sin(self.omega_0 * self.linear_1(self.weight_1*input))
        sine_2 = torch.sin(self.omega_0 * self.linear_2(sine_1))
        return self.weight_2*(input+sine_2)
    
    def change_nodes_per_layer(self, num_nodes):
        l1_weights = self.linear_1.weight.detach().clone()
        l1_bias = self.linear_1.bias.detach().clone()
        l2_weights = self.linear_2.weight.detach().clone()
        l2_bias = self.linear_2.bias.detach().clone()
        
        print(l1_weights.shape)
        self.features = num_nodes
        
        self.linear_1 = nn.Linear(num_nodes, num_nodes, bias=True)
        self.linear_2 = nn.Linear(num_nodes, num_nodes, bias=True)

        self.init_weights()
        
        print(self.linear_1.weight.shape)
    
class fSRN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.opt = opt
        self.net = []
        # if positional encoding is used, uncomment
        #self.net.append(
        #    PositionalEncoding(opt)
        #)

        # if positional encoding is used, this is the size for 
        # inputs to the first FC layer
        #opt['num_positional_encoding_terms']*opt['n_dims']*2, 
        self.net.append(
            SineLayer(
                opt['n_dims'],
                opt['nodes_per_layer'], 
                is_first=True, omega_0=opt['omega']
                )
            )

        i = 0
        while i < int(opt['n_layers']/2):
            self.net.append(ResidualSineLayer(opt['nodes_per_layer'], 
                ave_first=i>0,
                ave_second=(i==opt['n_layers']-1),
                omega_0=opt['omega']))                 
            i += 1

        final_linear = nn.Linear(opt['nodes_per_layer'], 
                                 opt['n_outputs'], bias=True)
            
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / opt['nodes_per_layer']) / 30, 
                                            np.sqrt(6 / opt['nodes_per_layer']) / 30)
            
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
    
    def forward_potentials(self, coords):
        output = self.net(coords)

        # We make first scalar the scalar potential
        scalar_potential = output[:,0]
        # We make the last 3 values the vector potential
        vector_potential = output[:,1:4]
        return scalar_potential, vector_potential
    
    def forward_vectorfields(self, coords):
        output = self.net(coords)

        # We make first scalar the scalar potential
        scalar_potential = output[:,0]
        # We make the last 3 values the vector potential
        vector_potential = output[:,1:4]

        # Take the gradient of the scalar potential to 
        # generate the rotation free component of the HHD
        rotation_free = torch.autograd.grad(scalar_potential, 
            coords, grad_outputs=torch.ones_like(scalar_potential),
            create_graph=True
        )[0]
        #print(rotation_free.shape)

        # Take the curl of the vector potential to generate
        # the divergence free component of the HHD
        curl_components = []
        for i in range(vector_potential.shape[1]):
            divergence_free_i = torch.autograd.grad(vector_potential[:,i], 
                coords, grad_outputs=torch.ones_like(vector_potential[:,i]),
                create_graph=True
            )[0]
            #print(divergence_free_i.shape)
            curl_components.append(divergence_free_i)
        divergence_free = torch.stack(
            [
                curl_components[2][:,1] - curl_components[1][:,2],
                curl_components[0][:,2] - curl_components[2][:,0],
                curl_components[1][:,0] - curl_components[0][:,1]
            ],
            dim=1
        )
        
        return rotation_free, divergence_free
        
    def forward(self, coords):     
        output = self.net(coords)

        # We make first scalar the scalar potential
        scalar_potential = output[:,0]
        # We make the last 3 values the vector potential
        vector_potential = output[:,1:4]

        # Take the gradient of the scalar potential to 
        # generate the rotation free component of the HHD
        rotation_free = torch.autograd.grad(scalar_potential, 
            coords, grad_outputs=torch.ones_like(scalar_potential),
            create_graph=True
        )[0]
        #print(rotation_free.shape)

        # Take the curl of the vector potential to generate
        # the divergence free component of the HHD
        curl_components = []
        for i in range(vector_potential.shape[1]):
            divergence_free_i = torch.autograd.grad(vector_potential[:,i], 
                coords, grad_outputs=torch.ones_like(vector_potential[:,i]),
                create_graph=True
            )[0]
            #print(divergence_free_i.shape)
            curl_components.append(divergence_free_i)
        divergence_free = torch.stack(
            [
                curl_components[2][:,1] - curl_components[1][:,2],
                curl_components[0][:,2] - curl_components[2][:,0],
                curl_components[1][:,0] - curl_components[0][:,1]
            ],
            dim=1
        )
        
        model_output = rotation_free + divergence_free

        return model_output

        