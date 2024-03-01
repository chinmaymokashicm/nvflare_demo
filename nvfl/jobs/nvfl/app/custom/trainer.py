from network import Network
from pt_dataset import PTDataset

import yaml
import inspect
import os, gc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import make_model_learnable, model_learnable_to_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_opt.pt.model_persistence_format_manager import PTModelPersistenceFormatManager

torch.cuda.empty_cache()


def get_function_obj(func_name: str, filepath: str = "udf"):
    module = __import__(filepath)
    members = inspect.getmembers(module)
    functions = {name: obj for name, obj in members if inspect.isfunction(obj)}
    if func_name in functions:
        return functions[func_name]
    return Exception(f"Function {func_name} not found")


class Trainer(Executor):
    def __init__(self, data_loader_funcname: str, label_loader_funcname: str, transform_funcname: str, train_task_name=AppConstants.TASK_TRAIN, submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL, exclude_vars=None, pre_train_task_name=AppConstants.TASK_GET_WEIGHTS):

        super().__init__()

        with open("custom/vars.yaml", "r") as f:
            vars: dict = yaml.safe_load(f)
            self._nvfl_vars: dict = vars["nvfl"]
            self._train_vars: dict = vars["train"]
            self._pt_constants_vars: dict = vars["pt_constants"]

        self._lr = self._train_vars["lr"]
        self._epochs = self._train_vars["epochs"]
        self._batch_size = self._train_vars["batch_size"]
        self._train_task_name = train_task_name
        self._pre_train_task_name = pre_train_task_name
        self._submit_model_task_name = submit_model_task_name
        self._exclude_vars = exclude_vars

        self.model = Network()
        # self.device = torch.device(
        #     "cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"
        self.model.to(self.device)
        self._loss = getattr(nn, self._train_vars["loss_fn"])()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)

        data_loader = get_function_obj(
            data_loader_funcname) if data_loader_funcname is not None else None
        label_loader = get_function_obj(
            label_loader_funcname) if label_loader_funcname is not None else None
        transform = get_function_obj(
            transform_funcname) if transform_funcname is not None else None
        self._train_dataset = PTDataset(
            folderpath_root=self._nvfl_vars["data_subfolder"], train_or_test="train", data_loader=data_loader, label_loader=label_loader, transform=transform)
        self._train_loader = DataLoader(
            self._train_dataset, batch_size=self._batch_size, shuffle=True)
        self._n_iterations = len(self._train_dataset)

        # Setup the persistence manager to save PT model.
        # The default training configuration is used by persistence manager
        # in case no initial model is found.
        self._default_train_conf = {
            "train": {"model": type(self.model).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.model.state_dict(), default_train_conf=self._default_train_conf
        )

        self._pt_constants = vars["pt_constants"]

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        try:
            if task_name == self._pre_train_task_name:
                # Get the new state dict and send as weights
                return self._get_model_weights()
            elif task_name == self._train_task_name:
                # Get model weights
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(
                        fl_ctx, "Unable to extract dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_error(
                        fl_ctx, f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Convert weights to tensor. Run training
                torch_weights = {k: torch.as_tensor(
                    v) for k, v in dxo.data.items()}
                self._local_train(fl_ctx, torch_weights, abort_signal)
                # self.model.to("cpu")

                # Check the abort_signal after training.
                # local_train returns early if abort_signal is triggered.
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                # Save the local model after training.
                self._save_local_model(fl_ctx)

                # Get the new state dict and send as weights
                return self._get_model_weights()
            elif task_name == self._submit_model_task_name:
                # Load local model
                ml = self._load_local_model(fl_ctx)

                # Get the model parameters and create dxo from it
                dxo = model_learnable_to_dxo(ml)
                return dxo.to_shareable()
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            self.log_exception(fl_ctx, f"Exception in simple trainer: {e}.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _get_model_weights(self) -> Shareable:
        # Get the new state dict and send as weights
        weights = {k: v.cpu().numpy()
                   for k, v in self.model.state_dict().items()}

        outgoing_dxo = DXO(
            data_kind=DataKind.WEIGHTS, data=weights, meta={
                MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations}
        )
        return outgoing_dxo.to_shareable()

    def _local_train(self, fl_ctx, weights, abort_signal):
        scaler = torch.cuda.amp.GradScaler()
        # Set the model weights
        self.model.load_state_dict(state_dict=weights)

        # Basic training
        self.model.train()
        for epoch in range(self._epochs):
            running_loss = 0.0
            for i, batch in enumerate(self._train_loader):
                if abort_signal.triggered:
                    # If abort_signal is triggered, we simply return.
                    # The outside function will check it again and decide steps to take.
                    return

                self.optimizer.zero_grad()

                # with torch.autocast(device_type="cpu"):
                with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    # images, labels = batch[0].to(self.device), batch[1].to(self.device)
                    images, labels = batch[0].to(self.device, dtype=torch.float), batch[1].to(self.device)
                    # images, labels = batch[0].to(self.device, drype=torch.float), batch[1].to(self.device, dtype=torch.long)
                    # images, labels = batch[0], batch[1]
                    predictions = self.model(images)
                    # predictions = predictions.to(torch.long)
                    predictions_softmax = torch.nn.functional.softmax(predictions, dim=1)
                    labels = labels.to(torch.long)
                    # print(f"""
                        
                    #     loss: {self._loss}
                    #     predictions: {predictions.shape}
                    #     labels: {labels.shape}
                    #     types: predictions: {predictions.dtype}, labels: {labels.dtype}, predictions_softmax: {predictions_softmax.dtype}
                        
                    #     """)
                    cost = self._loss(predictions, labels)
                gc.collect()
                cost.backward()
                # scaler.scale(cost).backward()
                self.optimizer.step()
                # scaler.step(self.optimizer)
                # scaler.update()

                running_loss += cost.cpu().detach().numpy() / images.size()[0]
                if i % 3000 == 0:
                    self.log_info(
                        fl_ctx, f"Epoch: {epoch}/{self._epochs}, Iteration: {i}, " f"Loss: {running_loss/3000}"
                    )
                    running_loss = 0.0

    def _save_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(
            fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, self._pt_constants_vars["modelsdir"])
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_path = os.path.join(
            models_dir, self._pt_constants_vars["localmodelname"])

        ml = make_model_learnable(self.model.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), model_path)

    def _load_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(
            fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, self._pt_constants_vars["modelsdir"])
        if not os.path.exists(models_dir):
            return None
        model_path = os.path.join(
            models_dir, self._pt_constants_vars["localmodelname"])

        self.persistence_manager = PTModelPersistenceFormatManager(
            data=torch.load(model_path), default_train_conf=self._default_train_conf
        )
        ml = self.persistence_manager.to_model_learnable(
            exclude_vars=self._exclude_vars)
        return ml
