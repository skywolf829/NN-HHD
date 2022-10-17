# NN-HHD
Using neural networks for the Helmholtz-Hodge Decomposition of a 3D vector field.


## Installation

We use conda to manage packages for Python 3.9.

`conda env create -f environement.yml`
`conda activate NNHHD`

Once that is done, create a folder called "Data" and populate that with your NetCDF files, which should have the vector field data as a [3,z,y,x] shape array in the "data" attribute.

## Example code

We run code using the start_jobs.py script, which will parse a job settings JSON and issue one job to each available compute device (many GPUs, or a single CPU).

`python Code/start_jobs.py --settings train.json`

Afterward, models are saved in SavedModels. Models can be tested in a similar fashion:

`python Code/start_jobs.py --settings test.json`

## Tensorboard 

We support tensorboard for visualization of loss values during training. To visualize and compare loss values of models, run:

`tensorboard --logdir tensorboard`

in a terminal, and then navigate to https://localhost:6006 to view the graphs.