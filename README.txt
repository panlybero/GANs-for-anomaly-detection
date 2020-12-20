model.py -- includes model definitions. I tried a few architectures before settling on one that worked, that file includes some of those old ones as well.

utils.py -- includes some utilty functions as well as function for training and evaluating models.

train.ipynb -- Notebook to reproduce report results. Calls on model.py and utils.py

models -- Folder that has pretrained models used to obtain the results in the report. The train.py notebook shows how to load the models. 

Requirements:
A yml file of my conda environment is included in the submission, though it has extra/unused packages.
Crucial dependencies are pytorch, torchvision, pandas, numpy, matplotlib
Packages should be up to date but for For exact versions please refer to he environment file. 
