import os
import sys
import time
import numpy as np
sys.path.append(os.path.join(os.getcwd(), "..", "naturalHHD", "pynhhd-v1.1"))
from pynhhd import nHHD
import argparse 
from utility_functions import tensor_to_cdf, nc_to_tensor
import torch

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Uses a Poisson solver for the Helmholtz-Hodge decomposition.')

    parser.add_argument('--data',default=None,type=str,
        help='Number of dimensions in the data')
    args = vars(parser.parse_args())

    data = nc_to_tensor(os.path.join(data_folder, args['data']))
    print(f"Data {args['data']} shape {data.shape}")

    data = data.cpu().numpy()[0].transpose(1, 2, 3, 0)
    n_dims = data.shape[-1]
    nx = data.shape[0]
    ny = data.shape[1]
    nz = data.shape[2]

    dims = (nx, ny, nz)
    dx = (2/nx, 2/ny, 2/nz)

    nhhd = nHHD(grid=dims, spacings=dx)
    t0 = time.time()
    print(f"Starting HHD")
    nhhd.decompose(data)
    t1 = time.time()

    print(f"Time for decompose: {t1-t0 : 0.02f}")

    print(f"Saving potentials and gradients")

    scalar_potential = nhhd.nD
    vector_potential = np.stack([nhhd.nRu, nhhd.nRv, nhhd.nRw], axis=-1)

    rotation_free = nhhd.d
    divergence_free = nhhd.r
    harmonic = nhhd.h

    tensor_to_cdf(torch.tensor(scalar_potential).unsqueeze(0).unsqueeze(0),
        os.path.join(output_folder, args['data'].split(".nc")[0]+"_scalarpotential.nc"))
    tensor_to_cdf(torch.tensor(vector_potential).permute(3, 0, 1, 2).unsqueeze(0),
        os.path.join(output_folder, args['data'].split(".nc")[0]+"_vectorpotential.nc"))

    tensor_to_cdf(torch.tensor(rotation_free).permute(3, 0, 1, 2).unsqueeze(0),
        os.path.join(output_folder, args['data'].split(".nc")[0]+"_rotationfree.nc"))
    tensor_to_cdf(torch.tensor(divergence_free).permute(3, 0, 1, 2).unsqueeze(0),
        os.path.join(output_folder, args['data'].split(".nc")[0]+"_divergencefree.nc"))
    tensor_to_cdf(torch.tensor(harmonic).permute(3, 0, 1, 2).unsqueeze(0),
        os.path.join(output_folder, args['data'].split(".nc")[0]+"_harmonic.nc"))

    
