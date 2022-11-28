from __future__ import absolute_import, division, print_function
import argparse
import os
from datasets import generate_synthetic_field1

project_folder_path = os.path.dirname(os.path.abspath(__file__))
project_folder_path = os.path.join(project_folder_path, "..")
data_folder = os.path.join(project_folder_path, "Data")
output_folder = os.path.join(project_folder_path, "Output")
save_folder = os.path.join(project_folder_path, "SavedModels")

if __name__ == '__main__':
   generate_synthetic_field1()
    
        
    
        



        

