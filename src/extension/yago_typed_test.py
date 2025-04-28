"""
Author: Ivan Diliso
Description: Performance and usability evaluation of implemented negative samplers
"""


import pykeen
from pykeen.triples import TriplesFactory
from pathlib import Path
from tabulate import tabulate
from test_utils import *
from pykeen.pipeline import pipeline
from pykeen.sampling.filtering import PythonSetFilterer
from extended_filtering import NullPythonSetFilterer
from extended_sampling import CorruptNegativeSampler, TypedNegativeSampler, RelationalNegativeSampler
from extended_dataset import OnMemoryDataset
import torch
from datetime import datetime


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


# Testing Sampling Techniques
################################################################################
# sampler = TypedNegativeSampler(
#     mapped_triples=dataset.training.mapped_triples,
#     filtered=True,
#     filterer=NullPythonSetFilterer(mapped_triples=dataset.training.mapped_triples),
#     num_negs_per_pos = 5,
#     entity_classes_dict = dataset.entity_id_to_classes,
#     relation_domain_range_dict = dataset.relation_id_to_domain_range
# )


mapped_triples = torch.tensor([
    [0, 0, 1],
    [0, 0, 2],
    [0, 0, 3],
    [0, 0, 4],
    [1, 1, 0],
    [1, 0, 0],
    [1, 1, 2],
    [1, 1, 3],
    [2, 0, 3],
    [2, 0, 4]
])


mapped_triples = dataset.training.mapped_triples


print(mapped_triples[mapped_triples[:, 2] == 68057])


local_file = Path().cwd() / "relational_save.bin"



sampler = RelationalNegativeSampler(
    mapped_triples=mapped_triples,
    filtered=True,
    filterer=NullPythonSetFilterer(
        mapped_triples=mapped_triples
    ),
    num_negs_per_pos=2,
    local_file=local_file
)

#print(sampler.subset)

#print(dataset.relation_to_id)

#negatives = sampler.sample(dataset.training.mapped_triples[torch.randperm(len(dataset.training.mapped_triples))][:10])

# for k, v in sampler.subset.items():
#     print("----------------------------------------------")
    
#     print(f"For entity {k}")
#     print(mapped_triples[mapped_triples[:, 0] == k])
#     print("When it appears as HEAD i can take the following to corrupt its TAIL")
#     for rel, values in v["tail"].items():
#         print(f"{rel} {values}")

#     print("")
#     print(mapped_triples[mapped_triples[:, 2] == k])
#     print("When it appears as TAIL i can take the following to corrupt its HEAD")
#     for rel, values in v["head"].items():
#         print(f"{rel} {values}")

#print(negatives)

#print("Generating Negatives")

negatives = sampler.sample(mapped_triples[100:120])


print(negatives)






