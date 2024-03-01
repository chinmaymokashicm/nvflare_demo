from pt_dataset import PTDataset
from network import Network

import inspect, yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants

def get_function_obj(func_name: str, filepath: str = "udf"):
    module = __import__(filepath)
    members = inspect.getmembers(module)
    functions = {name: obj for name, obj in members if inspect.isfunction(obj)}
    if func_name in functions:
        return functions[func_name]
    return Exception(f"Function {func_name} not found")

class Validator(Executor):
    def __init__(self, data_loader_funcname: str, label_loader_funcname: str, transform_funcname: str, validate_task_name=AppConstants.TASK_VALIDATION):
        
        super().__init__()
        with open("custom/vars.yaml", "r") as f:
            vars: dict = yaml.safe_load(f)
            self._nvfl_vars: dict = vars["nvfl"]
            self._train_vars: dict = vars["train"]

        self._validate_task_name = validate_task_name

        # Setup the model
        self.model = Network()
        # self.device = torch.device(
        #     "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        
        data_loader = get_function_obj(
            data_loader_funcname) if data_loader_funcname is not None else None
        label_loader = get_function_obj(
            label_loader_funcname) if label_loader_funcname is not None else None
        transform = get_function_obj(
            transform_funcname) if transform_funcname is not None else None
        self._test_dataset = PTDataset(
            folderpath_root=self._nvfl_vars["data_subfolder"], train_or_test="test", data_loader=data_loader, label_loader=label_loader, transform=transform)
        self._test_loader = DataLoader(
            self._test_dataset, batch_size=4, shuffle=True)
        
    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == self._validate_task_name:
            model_owner = "?"
            try:
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Error in extracting dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data_kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_exception(fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Extract weights and ensure they are tensor.
                model_owner = shareable.get_header(AppConstants.MODEL_OWNER, "?")
                weights = {k: torch.as_tensor(v, device=self.device) for k, v in dxo.data.items()}

                # Get validation accuracy
                val_accuracy = self._validate(weights, abort_signal)
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(
                    fl_ctx,
                    f"Accuracy when validating {model_owner}'s model on"
                    f" {fl_ctx.get_identity_name()}"
                    f"s data: {val_accuracy}",
                )

                dxo = DXO(data_kind=DataKind.METRICS, data={"val_acc": val_accuracy})
                return dxo.to_shareable()
            except:
                self.log_exception(fl_ctx, f"Exception in validating model from {model_owner}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def _validate(self, weights, abort_signal):
        self.model.load_state_dict(weights)

        self.model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self._test_loader):
                if abort_signal.triggered:
                    return 0

                images, labels = images.to(self.device), labels.to(self.device)
                output = self.model(images)

                _, pred_label = torch.max(output, 1)

                correct += (pred_label == labels).sum().item()
                total += images.size()[0]

            metric = correct / float(total)

        return metric
