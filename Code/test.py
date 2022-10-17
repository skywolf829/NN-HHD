from __future__ import absolute_import, division, print_function
import argparse
import os
from utility_functions import PSNR, tensor_to_cdf, create_path, particle_tracing, make_coord_grid
from models import load_model, sample_grid, sample_grad_grid
from options import load_options
import torch.nn.functional as F
from datasets import Dataset
import torch
import time
import numpy as np

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def model_reconstruction(model, dataset, opt):
    grid = list(dataset.data.shape[2:])
    if("dsfm" in opt['training_mode']): 
        grads_f = sample_grad_grid(model, grid, output_dim=0, 
                                        max_points=10000)
        grads_g = sample_grad_grid(model, grid, output_dim=1, 
                                        max_points=10000)
        grads_f = grads_f.permute(3, 0, 1, 2).unsqueeze(0)
        grads_g = grads_g.permute(3, 0, 1, 2).unsqueeze(0)
        with torch.no_grad():
            m = sample_grid(model, grid, max_points=10000)[...,2:3]
            m = m.permute(3, 0, 1, 2).unsqueeze(0)
        result = torch.cross(grads_f, grads_g, dim=1)
        result /= (result.norm(dim=1) + 1e-8)
        result *= m
        
    elif("uvw" in opt['training_mode']):
        with torch.no_grad():
            result = sample_grid(model, grid, max_points = 10000)
            result = result[...,0:3]
            result = result.permute(3, 0, 1, 2).unsqueeze(0)
            
    result = result.to(opt['data_device'])

    p = PSNR(result, data.data)

    print(f"PSNR: {p : 0.02f}")
    create_path(os.path.join(output_folder, "Reconstruction"))
    tensor_to_cdf(result, os.path.join(output_folder, "Reconstruction", opt['save_name']+".nc"))

def perform_tests(model, data, tests, opt):
    if("reconstruction" in tests):
        model_reconstruction(model, data, opt)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a model on some tests')

    parser.add_argument('--load_from',default=None,type=str,help="Model name to load")
    parser.add_argument('--tests_to_run',default=None,type=str,
                        help="A set of tests to run, separated by commas")
    parser.add_argument('--device',default=None,type=str,
                        help="Device to load model to")
    parser.add_argument('--data_device',default=None,type=str,
                        help="Device to load data to")
    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..")
    data_folder = os.path.join(project_folder_path, "Data")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")
    
    tests_to_run = args['tests_to_run'].split(',')
    
    # Load the model
    opt = load_options(os.path.join(save_folder, args['load_from']))
    opt['device'] = args['device']
    opt['data_device'] = args['data_device']
    model = load_model(opt, args['device']).to(args['device'])
    model.eval()
    
    # Load the reference data
    data = Dataset(opt)
    
    # Perform tests
    perform_tests(model, data, tests_to_run, opt)
    
        
    
        



        

