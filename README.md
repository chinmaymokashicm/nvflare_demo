# NVFlare using Pytorch
A command-line interface that guides you through choosing your own PyTorch network, your own custom dataset for machine learning, and user-defined functions for data processing or transformation.

## Before running the app, ensure that you have the following-
- network.py file in the root folder, that describes the PyTorch network named "Network"
- udf.py file in the root folder, that would contain your custom UDFs for functions such as data loading, transformation, etc.
- Updated vars.yaml file that contains some of the configuration for your NVFlare application
- Updated project.yml file that contains config for provisioning resources for a production-level application


## Running the command-line-interface
- Preferably in a python virtual environment, install dependencies-
`
pip install -r requirements.txt
`
- Run the CLI-
`
python cli.py
`