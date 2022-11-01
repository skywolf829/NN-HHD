from __future__ import absolute_import, division, print_function
import argparse
import os
from utility_functions import PSNR, tensor_to_cdf, create_path
from models import load_model, sample_grid, sample_grid_potentials, sample_grid_vectorfields
from options import load_options
from datasets import Dataset

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

def model_reconstruction(model, dataset, opt):
    grid = list(dataset.data.shape[2:])
    result = sample_grid(model, grid, 100000)
    result = result.to(opt['data_device'])
    result = result.permute(3, 0, 1, 2).unsqueeze(0)
    p = PSNR(result, dataset.data)

    print(f"PSNR: {p : 0.02f}")
    create_path(os.path.join(output_folder, "Reconstruction"))
    tensor_to_cdf(result, os.path.join(output_folder, "Reconstruction", opt['save_name']+".nc"))

def model_potentials(model, dataset, opt):
    grid = list(dataset.data.shape[2:])
    sp, vp = sample_grid_potentials(model, grid, 100000)
    sp = sp.to(opt['data_device'])
    print(sp.shape)
    sp = sp.permute(3, 0, 1, 2).unsqueeze(0)
    vp = vp.to(opt['data_device'])
    vp = vp.permute(3, 0, 1, 2).unsqueeze(0)
    
    create_path(os.path.join(output_folder, "Potentials"))
    tensor_to_cdf(sp, os.path.join(output_folder, 
            "Potentials", opt['save_name']+"_scalarpotential.nc"))
    tensor_to_cdf(vp, os.path.join(output_folder, 
            "Potentials", opt['save_name']+"_vectorpotential.nc"))

def model_vectorfields(model, dataset, opt):
    grid = list(dataset.data.shape[2:])
    rotationfree, divergencefree = sample_grid_vectorfields(model, grid, 100000)

    rotationfree = rotationfree.to(opt['data_device'])
    rotationfree = rotationfree.permute(3, 0, 1, 2).unsqueeze(0)
    divergencefree = divergencefree.to(opt['data_device'])
    divergencefree = divergencefree.permute(3, 0, 1, 2).unsqueeze(0)
    
    create_path(os.path.join(output_folder, "VectorFields"))
    tensor_to_cdf(rotationfree, os.path.join(output_folder, 
            "VectorFields", opt['save_name']+"_rotationfree.nc"))
    tensor_to_cdf(divergencefree, os.path.join(output_folder, 
            "VectorFields", opt['save_name']+"_divergencefree.nc"))
    
def perform_tests(model, data, tests, opt):
    if("reconstruction" in tests):
        model_reconstruction(model, data, opt)
    if("potentials" in tests):
        model_potentials(model, data, opt)
    if("vectorfields" in tests):
        model_vectorfields(model, data, opt)
    
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
    
        
    
        



        

