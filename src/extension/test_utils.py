import pykeen
from pykeen.triples import TriplesFactory
import random
import numpy as np
import pykeen.utils
import torch
import torch_geometric
import pykeen
from datetime import datetime
from timeit import default_timer as timer


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
    torch_geometric.seed_everything(seed)

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


class SimpleLogger():
    def __init__(self):
        self.time = 0
        self.str = ""
    
    def start(self, str=""):
        self.str = str
        print(f"[{color.RED}START{color.END}] {str} ...", end="\r")
        self.time = timer()

    def end(self):
        self.time = timer() - self.time
        print(f"[{color.GREEN}DONE{color.END} ] {self.str} in {color.CYAN}{self.time:09.4f}{color.END}s", end="\n")

