import pykeen
from pykeen.triples import TriplesFactory
from pathlib import Path
from tabulate import tabulate
from test_utils import *
from pykeen.pipeline import pipeline
from pykeen.sampling.filtering import PythonSetFilterer
from extended_filtering import NullPythonSetFilterer
from extended_sampling import CorruptNegativeSampler
from extended_dataset import OnMemoryDataset


# Initial Global Variables Configuration 
################################################################################
config = {
    "home_path"     : Path().cwd(),
    "dataset_name"  : "YAGO4-20"
}
config["data_path"] = config["home_path"] / "data" / config["dataset_name"]


# Dataset Loading
################################################################################
dataset = OnMemoryDataset(config["data_path"])

pretty_print("t", "Dataset Information")
print(f"Statistics: {dataset.num_entities} entities, {dataset.num_relations} relations")
print(tabulate(
    [[dataset.training.num_triples, dataset.validation.num_triples, dataset.testing.num_triples]],
    headers=["Train", "Valid", "Test"],
    tablefmt="github"
))









