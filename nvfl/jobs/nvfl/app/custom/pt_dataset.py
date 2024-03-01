import os, random
from typing import Optional, Callable, Any, Tuple

from torch.utils.data import Dataset
import numpy as np

class PTDataset(Dataset):
    def __init__(self, folderpath_root: str, train_or_test: str, data_loader: Callable[[str], np.ndarray], label_loader: Optional[Callable[[str], Any]], is_label: bool = True, transform: Optional[Callable] = None) -> None:
        self.folderpath_root = folderpath_root
        self.train_or_test = train_or_test
        self.transform = transform
        self.data_loader = data_loader
        self.is_label = is_label
        if self.is_label:
            self.filepaths_labels: list[str] = [os.path.join(self.folderpath_root, train_or_test, "labels", filename) for filename in os.listdir(os.path.join(self.folderpath_root, train_or_test, "labels"))]
            self.label_loader = label_loader
            
        self.filepaths_data: list[str] = [os.path.join(self.folderpath_root, train_or_test, "data", filename) for filename in os.listdir(os.path.join(self.folderpath_root, train_or_test, "data"))]
        
    def __len__(self) -> int:
        return len(self.filepaths_data)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Any]:
        filepath_data = self.filepaths_data[idx]
        data = self.data_loader(filepath_data)
        
        if self.transform:
            # print(data.shape)
            data = self.transform(data)
            # print(data.shape)
        
        if self.is_label:
            filepath_label = self.filepaths_labels[idx]
            label = self.label_loader(filepath_label)
            # if self.transform:
            #     label = self.transform(label)
            # print(f"Data shape: {data.shape}, Label shape: {label.shape}")
            return data, label
        
        return data