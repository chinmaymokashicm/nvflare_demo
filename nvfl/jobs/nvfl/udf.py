import numpy as np
import nibabel as nib


def load_nifti(filepath_data: str) -> np.ndarray:
    img = nib.load(filepath_data)
    return img.get_fdata()

def test() -> None:
    pass