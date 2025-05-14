"""
Author: Ivan Diliso
Description: Performance and usability evaluation of implemented negative samplers
"""

import gc
import signal
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pykeen
import pykeen.models
import torch
from extension.dataset import OnMemoryDataset
from extension.filtering import NullPythonSetFilterer
from extension.sampling import (
    CorruptNegativeSampler,
    RelationalNegativeSampler,
    TypedNegativeSampler,
    NearestNeighbourNegativeSampler,
    NearMissNegativeSampler,
)

from pykeen.sampling import BernoulliNegativeSampler, BasicNegativeSampler
from pykeen.models import TransE
from pykeen.pipeline import pipeline
from pykeen.sampling.filtering import PythonSetFilterer
from pykeen.triples import TriplesFactory
from tabulate import tabulate
from extension.utils import *

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
dataset_name = "FB15K"
config = {"home_path": Path().cwd(), "dataset_name": dataset_name}
config["data_path"] = config["home_path"] / "data" / config["dataset_name"]


# Dataset Loading
################################################################################
dataset = OnMemoryDataset(
    data_path=config["data_path"], load_domain_range=False, load_entity_classes=False
)

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
    weights_only=False,
)

sampling_model = sampling_model.to(torch.device("cpu"))


def sampling_model_prediction(model, hrt_batch, targets):
    out = torch.zeros(
        (hrt_batch.size(0), model.entity_representations[0]().size(1)),
        device=hrt_batch.device,
    )
    out[targets == 0] = model.entity_representations[0](
        hrt_batch[targets == 0, 2]
    ) - model.relation_representations[0](hrt_batch[targets == 0, 1])
    out[targets == 2] = model.entity_representations[0](
        hrt_batch[targets == 2, 0]
    ) + model.relation_representations[0](hrt_batch[targets == 2, 1])

    return out


print(sampling_model)


# Negative Samplers Setup
################################################################################

params = SimpleNamespace()


params.negative_sampler_name = "corrupt"
params.local_file = Path().cwd() / (params.negative_sampler_name + dataset_name + ".bin")
params.num_negs_per_pos = 100
params.sample = True
params.sample_size = 500
params.permutate_triples = True


if params.permutate_triples:
    eval_triples = dataset.training.mapped_triples[torch.randperm(len(dataset.training.mapped_triples))[:params.sample_size]]
else:
    eval_triples = dataset.training.mapped_triples[:params.sample_size]


match params.negative_sampler_name:
    case "random":
        params.negative_sampler = (
            BasicNegativeSampler(
                mapped_triples=dataset.training.mapped_triples,
                filtered=True,
                filterer=NullPythonSetFilterer(
                    mapped_triples=dataset.training.mapped_triples
                ),
                num_negs_per_pos=params.num_negs_per_pos,
            )
        )
    case "bernoulli":
        params.negative_sampler = (
            BernoulliNegativeSampler(
                mapped_triples=dataset.training.mapped_triples,
                filtered=True,
                filterer=NullPythonSetFilterer(
                    mapped_triples=dataset.training.mapped_triples
                ),
                num_negs_per_pos=params.num_negs_per_pos,
            )
        )
    case "corrupt":
        params.negative_sampler = (
            CorruptNegativeSampler(
                mapped_triples=dataset.training.mapped_triples,
                filtered=True,
                filterer=NullPythonSetFilterer(
                    mapped_triples=dataset.training.mapped_triples
                ),
                num_negs_per_pos=params.num_negs_per_pos,
            )
        )
    case "typed":
        params.negative_sampler = (
            TypedNegativeSampler(
                mapped_triples=dataset.training.mapped_triples,
                filtered=True,
                filterer=NullPythonSetFilterer(
                    mapped_triples=dataset.training.mapped_triples
                ),
                num_negs_per_pos=params.num_negs_per_pos,
                entity_classes_dict=dataset.entity_id_to_classes,
                relation_domain_range_dict=dataset.relation_id_to_domain_range,
            )
        )
    case "relational":
        params.negative_sampler = RelationalNegativeSampler(
            mapped_triples=dataset.training.mapped_triples,
            filtered=True,
            filterer=NullPythonSetFilterer(
                mapped_triples=dataset.training.mapped_triples
            ),
            num_negs_per_pos=params.num_negs_per_pos,
            local_file=params.local_file
        )



val, dict = params.negative_sampler.average_pool_size(dataset.training.mapped_triples)

print(f"Average Pool Size{val}")
for k,v in dict.items():
    print(k,v)


print("Sample Predictions")
if params.sample:
    log = SimpleLogger()

    log.start()
    negatives = params.negative_sampler.sample(eval_triples)
    log.end()

    print(negatives)
    print(negatives[1].shape)
    print(torch.sum(negatives[1]))
