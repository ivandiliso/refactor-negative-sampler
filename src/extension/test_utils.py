import pykeen
from pykeen.triples import TriplesFactory
import random
import numpy as np
import pykeen.utils
import torch
import pykeen
from datetime import datetime
from timeit import default_timer as timer


def automatic_backend_chooser() -> torch.device:
    """Automatically select the device for the current hardware

    Args:
        logger (util.logger, optional): Logging utility. Defaults to None.

    Returns:
        torch.device: Selected torch device
    """
    device = torch.device("cpu")

    if not torch.cuda.is_available():
        print("[Backend Chooser] CUDA NOT Available")
    else:
        print("[Backend Chooser] CUDA Available")
        device = torch.device("cuda")

    """
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "[Backend Chooser] MPS not available because the current PyTorch install was not built with MPS enabled."
            )
        else:

            print(
                "[Backend Chooser] MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine."
            )
    
    else:

        print("[Backend Chooser] MPS Available")
        device = torch.device("mps")
    """

    print(f"[Backend Chooser] Using {str(device).upper()} Acceleration")

    return device


def set_random_seed_all(seed: int) -> None:
    """Set random seed for all CUDA and tensor libraries for reproducibility

    Args:
        seed (int): Random seed
        logger (util.logger, optional): Logging utility. Defaults to None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if torch.mps.is_available():
        torch.mps.manual_seed(seed)

    pykeen.utils.set_random_seed(seed)


class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def pretty_print(type, str):
    match type:
        case "t":
            print(f"[{color.RED}{str}{color.END}]")


class SimpleLogger:
    def __init__(self):
        self.time = 0
        self.str = ""

    def start(self, str=""):
        self.str = str
        print(f"[{color.RED}START{color.END}] {str} ...", end="\r")
        self.time = timer()

    def end(self):
        self.time = timer() - self.time
        print(
            f"[{color.GREEN}DONE{color.END} ] {self.str} in {color.CYAN}{self.time:09.4f}{color.END}s",
            end="\n",
        )
