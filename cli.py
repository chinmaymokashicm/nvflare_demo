from nvfl.jobs.nvfl.app.custom.network_config import NetworkCheck
from nvfl.jobs.nvfl.app.custom.dataset import DatasetMigration, DatasetProps

import yaml, os, time, shutil, json

import typer, inspect
from typing import List, Any, Literal
from rich.console import Console
from yaspin import yaspin
import inquirer
from inquirer import prompt
from nvflare.private.fed.app.simulator.simulator_runner import SimulatorRunner

app = typer.Typer()
console = Console()

def load_yaml(filepath: str = "vars.yaml") -> dict:
    with open(filepath, "r") as f:
        return yaml.safe_load(f)
    
def load_json(filepath: str) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)
    
def get_functions(filename: str = "udf") -> list:
    filename = "udf"
    module = __import__(filename)
    members = inspect.getmembers(module)
    functions = [name for name, obj in members if inspect.isfunction(obj)]
    return functions
    
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
        
    console.print("Checking NN functioning")
    n_channels: int = int(typer.prompt("Provide number of channels"))
    input_dims_comma: str = typer.prompt("Provide input dimensions (comma-separated integers)")
    input_dims: list = [int(dim) for dim in input_dims_comma.split(",")]
    network_check = NetworkCheck(network=network, n_channels=n_channels, input_dims=input_dims)
    console.print(network_check)
    
def file_checks(nvfl_vars: dict) -> None:
    networkfile: str = nvfl_vars["networkfile"]
    udffile: str = nvfl_vars["udffile"]
    requirementsfile: str = "requirements.txt"
    folderpath: str = nvfl_vars["folderpath"]
    wait_for_files([networkfile, udffile, requirementsfile])
    test_nn_functionality(networkfile)
    shutil.copy2(networkfile, os.path.join(folderpath, "app", "custom"))
    shutil.copy2(udffile, os.path.join(folderpath, "app", "custom"))
    shutil.copy2("vars.yaml", os.path.join(folderpath, "app", "custom"))
    
def migrate_dataset(nvfl_vars: dict, one2one_mapping: bool = True) -> None:
    folderpath_root_source: str = typer.prompt("Enter folder path of the dataset")
    destination: str = os.path.join(nvfl_vars["folderpath"], "app", nvfl_vars["data_subfolder"])
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
    
def prepare_config(nvfl_vars: dict) -> None:
    config_fed_client_filepath: str = os.path.join(nvfl_vars["folderpath"], "app", "config", "config_fed_client.json")
    config_fed_client = load_json(config_fed_client_filepath)
    # config_fed_server_filepath: str = os.path.join(nvfl_vars["folderpath"], "app", "config", "config_fed_server.json")
    # config_fed_server = load_json(config_fed_server_filepath)
    
    functions: list = get_functions()
    placeholder_none: str = "None"
    choices: list = functions + [placeholder_none]
    categories: list = ["data_loader", "label_loader", "transform"]
    for category in categories:
        # Add questions here and update config accordingly
        question: list = [
            inquirer.List(f"{category} loader",
                            message=f"Choose {category} function",
                            choices=choices,
                        ),
            ]
        func_name: str = list(inquirer.prompt(question).values())[0]
        for i in range(len(config_fed_client["executors"])):
            config_fed_client["executors"][i]["executor"]["args"][f"{category}_funcname"] = func_name if func_name != placeholder_none else None
            
    with open(config_fed_client_filepath, "w") as f:
        json.dump(config_fed_client, f, indent=4)
        
def run_simulation(args: dict) -> Any | Literal[-9] | None:
    choices: dict = {
        "Yes": True,
        "No": False
    }
    q_run_simulation: list = [
    inquirer.List("run simulation",
                    message="Run simulation",
                    choices=choices.keys(),
                ),
    ]
    choice: bool = choices[list(inquirer.prompt(q_run_simulation).values())[0]]
    if choice:
        status = run_simulator(args)
        return status
        
def run_simulator(args) -> Any | Literal[-9]:
    simulator = SimulatorRunner(**args)
    run_status = simulator.run()
    return run_status

@app.command()
def main():
    vars: dict = load_yaml()
    nvfl_config: dict = vars["nvfl"]
    simulation_config: dict = vars["simulation"]
    file_checks(nvfl_config)
    migrate_dataset(nvfl_config, one2one_mapping=False)
    prepare_config(nvfl_config)
    run_status: Any | Literal = run_simulation(simulation_config)
    console.print(run_status)
    
if __name__=="__main__":
    app()