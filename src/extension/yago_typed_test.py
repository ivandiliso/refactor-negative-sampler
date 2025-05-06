"""
Author: Ivan Diliso
Description: Performance and usability evaluation of implemented negative samplers
"""

import gc
import signal
import sys
from datetime import datetime
from pathlib import Path

import pykeen
import pykeen.models
import torch
from extended_dataset import OnMemoryDataset
from extended_filtering import NullPythonSetFilterer
from extended_sampling import (
    CorruptNegativeSampler,
    RelationalNegativeSampler,
    TypedNegativeSampler,
    NearestNeighbourNegativeSampler,
    NearMissNegativeSampler
)
from pykeen.models import TransE
from pykeen.pipeline import pipeline
from pykeen.sampling.filtering import PythonSetFilterer
from pykeen.triples import TriplesFactory
from tabulate import tabulate
from test_utils import *

# Python and Torch Configuration
################################################################################

set_random_seed_all(42)

torch.cuda.empty_cache()


def cleanup():
    print("\n Cleaning up GPU memory")
    gc.collect()
    torch.cuda.empty_cache()
    print("Cleanup Complete")


def signal_handler(sig, frame):
    print("Exiting gracefully")
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


# Initial Global Variables Configuration
################################################################################
config = {"home_path": Path().cwd(), "dataset_name": "YAGO4-20"}
config["data_path"] = config["home_path"] / "data" / config["dataset_name"]


# Dataset Loading
################################################################################
dataset = OnMemoryDataset(config["data_path"])

pretty_print("t", "Dataset Information")
print(f"Statistics: {dataset.num_entities} entities, {dataset.num_relations} relations")
print(
    tabulate(
        [
            [
                dataset.training.num_triples,
                dataset.validation.num_triples,
                dataset.testing.num_triples,
            ]
        ],
        headers=["Train", "Valid", "Test"],
        tablefmt="github",
    )
)

# Loading Pretrained Model for Dynamic Sampling
################################################################################

sampling_model = torch.load(
    Path.cwd() / "model" / "sampling" / "transe_yago420" / "trained_model.pkl",
    weights_only=False
)

sampling_model = sampling_model.to(torch.device("mps"))

def sampling_model_prediction(model, hrt_batch, targets):

    out = torch.zeros((hrt_batch.size(0), model.entity_representations[0]().size(1)), device=hrt_batch.device)
    
    # Head
    out[targets == 0] = model.entity_representations[0](hrt_batch[targets == 0, 2]) - model.relation_representations[0](hrt_batch[targets == 0, 1])

    # Tails
    out[targets == 2] = model.entity_representations[0](hrt_batch[targets == 2, 0]) + model.relation_representations[0](hrt_batch[targets == 2, 1])

    return out



print(sampling_model)


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


mapped_triples = torch.tensor(
    [
        [0, 0, 1],
        [0, 0, 2],
        [0, 0, 3],
        [0, 0, 4],
        [1, 1, 0],
        [1, 0, 0],
        [1, 1, 2],
        [1, 1, 3],
        [2, 0, 3],
        [2, 0, 4],
    ], device=torch.device("mps")
)

print(sampling_model_prediction(sampling_model, mapped_triples, torch.tensor([0,2,2,2,0,0,0,2,0,2], device=torch.device("mps"))))


mapped_triples = dataset.training.mapped_triples
mapped_triples = mapped_triples.to(torch.device("mps"))

#print(mapped_triples[mapped_triples[:, 2] == 68057])


local_file = Path().cwd() / "nn_save.bin"

print(local_file)


sampler = NearMissNegativeSampler(
    mapped_triples=mapped_triples,
    filtered=True,
    filterer=NullPythonSetFilterer(mapped_triples=mapped_triples),
    local_file=local_file,
    sampling_model=sampling_model,
    num_negs_per_pos=100,
    prediction_function=sampling_model_prediction,
    batch_size=1024,
    device=torch.device("mps")
)

# print(sampler.subset)

# print(dataset.relation_to_id)

# negatives = sampler.sample(dataset.training.mapped_triples[torch.randperm(len(dataset.training.mapped_triples))][:10])

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

# print(negatives)

# print("Generating Negatives")

logger = SimpleLogger()


logger.start("SAMPLING BATCH")

negatives = sampler.sample(mapped_triples[:4096])

logger.end()



print(negatives)
print(negatives[1].shape)
print(torch.sum(negatives[1]))