from typing import Any, Tuple

import numpy as np
import nibabel as nib
import torch
from torchvision import transforms


def load_nifti(filepath_data: str) -> np.ndarray:
    img = nib.load(filepath_data)
    return img.get_fdata()

def test() -> None:
    pass

def transpose(array: np.ndarray) -> np.ndarray:
    return np.transpose(array, (3, 0, 1, 2))