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
from extended_sampling_optimized import (
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
dataset = OnMemoryDataset(
    data_path=config["data_path"], load_domain_range=True, load_entity_classes=True
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

mapped_triples = dataset.training.mapped_triples
LOCAL_FILE = Path().cwd() / "nn_save.bin"
NUM_NEG_PER_POS = 100
NUM_QUERY_RESULTS = 200
filterer = NullPythonSetFilterer(mapped_triples=mapped_triples)


negative_samplers = {
    "basic" : BasicNegativeSampler(
        mapped_triples=mapped_triples,
        filtered=True,
        filterer=filterer,
        num_neg_per_pos=NUM_NEG_PER_POS
    ),
    "bernoulli" : BernoulliNegativeSampler(
        mapped_triples=mapped_triples,
        filtered=True,
        filterer=filterer,
        num_neg_per_pos=NUM_NEG_PER_POS
    ),
    "corrupt" : CorruptNegativeSampler(
        mapped_triples=mapped_triples,
        filtered=True,
        filterer=filterer,
        num_neg_per_pos=NUM_NEG_PER_POS
    ),
    "typed" : TypedNegativeSampler(
        mapped_triples=mapped_triples,
        filtered=True,
        filterer=filterer,
        num_neg_per_pos=NUM_NEG_PER_POS,
        entity_classes_dict=dataset.entity_id_to_classes,
        relation_domain_range_dict=dataset.relation_id_to_domain_range
    ),
    "relational" : RelationalNegativeSampler(
        mapped_triples=mapped_triples,
        filtered=True,
        filterer=filterer,
        num_neg_per_pos=NUM_NEG_PER_POS,
        local_file=LOCAL_FILE
    ),
    "nearestneighbour" : NearestNeighbourNegativeSampler(
        mapped_triples=mapped_triples,
        filtered=True,
        filterer=filterer,
        num_neg_per_pos=NUM_NEG_PER_POS,
        sampling_model=sampling_model,
        num_query_results=NUM_QUERY_RESULTS
    ),
    "nearmiss" : NearMissNegativeSampler(
        mapped_triples=mapped_triples,
        filtered=True,
        filterer=filterer,
        num_neg_per_pos=NUM_NEG_PER_POS,
        num_query_results=NUM_QUERY_RESULTS,
        sampling_model=sampling_model,
        num_query_results=NUM_QUERY_RESULTS,
        prediction_function=sampling_model_prediction

    )
}








sampler = NearMissNegativeSampler(
    mapped_triples=mapped_triples,
    local_file=local_file,
    filtered=True,
    filterer=NullPythonSetFilterer(mapped_triples=mapped_triples),
    sampling_model=sampling_model,
    num_negs_per_pos=50,
    prediction_function=sampling_model_prediction,
    batch_size=1024,
    device=torch.device("cpu"),
    num_query_results=200,
)


log = SimpleLogger()

log.start()
negatives = sampler.sample(mapped_triples[:2048])
log.end()


print(negatives)
print(negatives[1].shape)
print(torch.sum(negatives[1]))
