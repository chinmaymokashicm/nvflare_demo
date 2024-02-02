from nvfl.jobs.nvfl.app.custom.network_config import NetworkCheck

import yaml, os, time

import typer
from typing import List
from rich.console import Console

app = typer.Typer()
console = Console()

def load_yaml(filepath: str = "vars.yaml") -> dict:
    with open(filepath, "r") as f:
        return yaml.safe_load(f)
    
def wait_for_files(files: List[str], timeout: int = 5):
    if not files:
        raise ValueError("'files' is empty")
    if not isinstance(files, list):
        raise TypeError("'files' must be a list")
    
    start_time = time.time()
    found_files = []

    console.print(f"Finding files: {files}")
    while time.time() - start_time < timeout:
        found_files = [file for file in files if os.path.exists(file)]

        if len(found_files) == len(files):
            console.print("All files found!")
            break
        console.print(f"Found: {found_files}. Remaining time: {int(timeout - (time.time() - start_time))}", end="\r")

        time.sleep(1)
        
    missing_files = set(files) - set(found_files)
    if len(missing_files) > 0:
        raise FileNotFoundError(f"Timeout reached. Files not found: {missing_files}")
        
def import_this(name: str) -> type:
    components = name.split(".")
    path = components[0].replace("/", ".").replace("\\", ".")
    module = __import__(path)
    for component in path.split(".")[1:]:
        module = getattr(module, component)
    return module
        
def test_nn_functionality(networkfile: str) -> None:
    console.print(f"Checking for network file {networkfile}...")
    try:
        module = import_this(networkfile)
        classname = "Network"
        network_class = getattr(module, classname)
        network = network_class()
        console.print(f"Found {networkfile}.{classname}")
    except AttributeError:
        console.print(f"Could not find network class!")
        
    input_dims_comma: str = typer.prompt("Step 1: Checking NN functioning. Provide input dimensions (comma-separated integers)")
    input_dims: list = [int(dim) for dim in input_dims_comma.split(",")]
    network_check = NetworkCheck(network=network, input_dims=input_dims)
    console.print(network_check)

@app.command()
def main():
    vars: dict = load_yaml()
    networkfile: str = vars["networkfile"]
    udffile: str = vars["udffile"]
    requirementsfile: str = "requirements.txt"
    wait_for_files([networkfile, udffile, requirementsfile])
    test_nn_functionality(networkfile)
    
    nvfl_folderpath: str = vars["nvfl_folderpath"]
    
if __name__=="__main__":
    app()