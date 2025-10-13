# encoding: utf-8

import numpy as np
import torch
import random
import os


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility across numpy, torch, and python.
    Also ensures determinism in torch backend (cudnn).

    Args:
        seed (int): the seed value to use. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Debug/test
if __name__ == '__main__':
    set_random_seed(0)
    print("Random float from NumPy:", np.random.random())
    print("Random tensor from PyTorch:", torch.rand(1))
