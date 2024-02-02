from nvfl.jobs.nvfl.app.custom.network_config import NetworkCheck
from nvfl.jobs.nvfl.app.custom.dataset import DatasetMigration, DatasetProps

import yaml, os, time, shutil

import typer
from typing import List
from rich.console import Console
from yaspin import yaspin
import inquirer
from inquirer import prompt

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
    
    with yaspin():
        while time.time() - start_time < timeout:
            found_files = [file for file in files if os.path.exists(file)]

            if len(found_files) == len(files):
                console.print("All files found!")
                break
            console.print(f"Found: {found_files}. Remaining time: {int(timeout - (time.time() - start_time))}", end="\r")

            with yaspin():
                time.sleep(5)
        
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
    
def file_checks(vars: dict) -> None:
    networkfile: str = vars["networkfile"]
    udffile: str = vars["udffile"]
    requirementsfile: str = "requirements.txt"
    nvfl_folderpath: str = vars["nvfl_folderpath"]
    wait_for_files([networkfile, udffile, requirementsfile])
    test_nn_functionality(networkfile)
    shutil.copy2(networkfile, os.path.join(nvfl_folderpath, "app", "custom"))
    shutil.copy2(udffile, os.path.join(nvfl_folderpath, "app", "custom"))
    
def migrate_dataset(vars: dict, one2one_mapping: bool = True) -> None:
    folderpath_root_source: str = typer.prompt("Enter folder path of the dataset")
    destination: str = os.path.join(vars["nvfl_folderpath"], vars["data_subfolder"])
    dataset_migration: DatasetMigration = DatasetMigration(folderpath_root_source, folderpath_destination_root=destination)
    remaining_subfolders_source: list = [subfolder for subfolder in os.listdir(folderpath_root_source) if os.path.isdir(os.path.join(folderpath_root_source, subfolder))]
    subfolders_destination: list = dataset_migration.folderpath_mappings
    for subfolder_destination in subfolders_destination:
        q_folderpath_mappings: list = [
            inquirer.List("folder mappings",
                        message = f"Choose folder for {subfolder_destination}",
                        choices = remaining_subfolders_source
                        )
        ]
        subfolder_source = list(inquirer.prompt(q_folderpath_mappings).values())[0]
        dataset_migration.add_foldermap(subfolder_destination, subfolder_source)
        if one2one_mapping:
            remaining_subfolders_source.remove(subfolder_source)
    dataset_migration.create_filepath_mappings()
    dataset_migration.migrate_data(make_symlink=True)

@app.command()
def main():
    vars: dict = load_yaml()
    file_checks(vars)
    migrate_dataset(vars, one2one_mapping=False)
    
if __name__=="__main__":
    app()